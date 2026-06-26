"""live/inference.py — Pi側: ONNX Runtime 推論.

特徴量は data/features.py（Polars）を**そのまま再利用**して計算する（train/live を
同一コードで構造的に一致させる）。正規化は ONNX グラフに同梱（NormalizedModel）
されているため、ここでは生特徴量の窓をそのまま渡す。

フロー（1バーごと）:
  recent buffer(DB) → features.compute_features（ローリング窓・末尾を使用）
    → 末尾 seq_len 行の生特徴量 (seq_len, 58) → ONNX → softmax
    → signal/確率/σ → model_predictions へ書き込み

ウォームアップ: features.WARMUP_BARS + seq_len バー以上の履歴を渡すこと
（末尾 seq_len 行をバルク計算と一致させるため）。

onnxruntime / asyncpg は遅延 import。純粋な部分（softmax / build_window /
predict）は単体テストできる。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from data.features import WARMUP_BARS, compute_features, fetch_book, fetch_bars, fetch_funding
from model.dataset import CLASS_TO_LABEL
from shared.feature_names import FEATURE_NAMES, SIGMA_FEATURE

log = logging.getLogger("aevum.inference")

DEFAULT_SEQ_LEN = 128

MODEL_PREDICTIONS_INSERT_SQL = (
    "INSERT INTO model_predictions "
    "(symbol, time, model_version, prob_down, prob_flat, prob_up, signal, sigma) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
    "ON CONFLICT (symbol, time, model_version) DO UPDATE SET "
    "prob_down = EXCLUDED.prob_down, prob_flat = EXCLUDED.prob_flat, "
    "prob_up = EXCLUDED.prob_up, signal = EXCLUDED.signal, sigma = EXCLUDED.sigma"
)


# --------------------------------------------------------------------------- #
# 純粋関数
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def build_window(features_df, seq_len: int, feature_names=FEATURE_NAMES) -> Optional[np.ndarray]:
    """末尾 seq_len 行の生特徴量 (seq_len, n_features) を返す。

    行数不足、または末尾窓に NaN（ウォームアップ未完）があれば None（推論不可）。
    列順は FEATURE_NAMES（features.py が保証する単一の真実）。
    """
    if features_df.height < seq_len:
        return None
    arr = features_df.tail(seq_len).select(list(feature_names)).to_numpy().astype(np.float32)
    if not np.isfinite(arr).all():
        return None
    return arr


# --------------------------------------------------------------------------- #
# ONNX 予測器
# --------------------------------------------------------------------------- #
class OnnxPredictor:
    """ONNX(正規化同梱) をロードし、生特徴量窓から確率・シグナルを返す。"""

    def __init__(self, onnx_path, metadata_path=None, *, providers=None, model_version=None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        meta = {}
        if metadata_path is not None and Path(metadata_path).is_file():
            meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.seq_len = int(meta.get("seq_len", DEFAULT_SEQ_LEN))
        self.feature_names = tuple(meta.get("feature_names", FEATURE_NAMES))
        self.model_version = model_version or meta.get("model_version") or Path(onnx_path).stem

    def predict_logits(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        return self.session.run([self.output_name], {self.input_name: x})[0]

    def predict(self, window: np.ndarray) -> dict:
        """単一窓 → {probs[3], class, signal}。probs=[down(-1), flat(0), up(+1)]。"""
        probs = softmax(self.predict_logits(window))[0]
        cls = int(probs.argmax())
        return {"probs": probs, "class": cls, "signal": CLASS_TO_LABEL[cls]}


# --------------------------------------------------------------------------- #
# 推論（特徴量計算 → 予測レコード）
# --------------------------------------------------------------------------- #
def infer_symbol(
    predictor: OnnxPredictor,
    symbol: str,
    bars_by_sym: dict,
    book_by_sym: Optional[dict] = None,
    funding_by_sym: Optional[dict] = None,
) -> Optional[dict]:
    """指定銘柄の最新バーに対する予測レコードを返す（窓が未確定なら None）。"""
    feats = compute_features(bars_by_sym, book_by_sym, funding_by_sym)
    df = feats.get(symbol)
    if df is None or df.height == 0:
        return None
    window = build_window(df, predictor.seq_len, predictor.feature_names)
    if window is None:
        return None
    pred = predictor.predict(window)
    last = df.tail(1)
    p = pred["probs"]
    return {
        "symbol": symbol,
        "time": last["time"][0],
        "model_version": predictor.model_version,
        "prob_down": float(p[0]),
        "prob_flat": float(p[1]),
        "prob_up": float(p[2]),
        "signal": int(pred["signal"]),
        "sigma": float(last[SIGMA_FEATURE][0]),
    }


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
async def store_prediction(conn, rec: dict) -> None:
    await conn.execute(
        MODEL_PREDICTIONS_INSERT_SQL,
        rec["symbol"], rec["time"], rec["model_version"],
        rec["prob_down"], rec["prob_flat"], rec["prob_up"],
        rec["signal"], rec["sigma"],
    )


async def _fetch_recent(conn, symbol, predictor):
    """直近 WARMUP_BARS + seq_len バーぶんのバッファを取得（bars/book/funding）。

    板は計算負荷が大きいが、窓(seq_len)+spread rolling(60) ぶんしか使われないため
    直近 seq_len+64 本に限定する（古い板は窓に入らない）。bars/funding は履歴依存の
    特徴量があるため WARMUP 全体を渡す。
    """
    n = WARMUP_BARS + predictor.seq_len
    bars = (await fetch_bars(conn, symbol)).tail(n)
    book = await fetch_book(conn, symbol)
    funding = await fetch_funding(conn, symbol)
    if bars.height:
        bar_start = bars["time"][0]
        book_bars = min(bars.height, predictor.seq_len + 64)
        book_start = bars["time"][bars.height - book_bars]
        if book.height:
            book = book.filter(book["time"] >= book_start)
        if funding.height:
            funding = funding.filter(funding["time"] >= bar_start)
    return bars, book, funding


async def run_once(conn, predictor: OnnxPredictor, symbols=("BTC", "ETH")) -> list[dict]:
    """各銘柄の最新バーで推論し model_predictions に書き込む。書いたレコードを返す。"""
    bars_by_sym, book_by_sym, funding_by_sym = {}, {}, {}
    for s in symbols:
        bars, book, funding = await _fetch_recent(conn, s, predictor)
        bars_by_sym[s], book_by_sym[s], funding_by_sym[s] = bars, book, funding

    written = []
    for s in symbols:
        rec = infer_symbol(predictor, s, bars_by_sym, book_by_sym, funding_by_sym)
        if rec is not None:
            await store_prediction(conn, rec)
            written.append(rec)
            log.info("prediction: %s signal=%d p=(%.3f,%.3f,%.3f)",
                     s, rec["signal"], rec["prob_down"], rec["prob_flat"], rec["prob_up"])
    return written


async def run(onnx_path="artifacts/model.onnx", metadata_path="artifacts/metadata.json",
              symbols=("BTC", "ETH")) -> None:
    import asyncpg

    from data.ingestion import resolve_dsn

    predictor = OnnxPredictor(onnx_path, metadata_path)
    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await run_once(conn, predictor, symbols)
    finally:
        await pool.close()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
