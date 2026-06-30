"""live/inference.py の単体テスト。

softmax / build_window は torch/onnx 非依存。OnnxPredictor / infer_symbol は
小さな ONNX を実エクスポートして検証（torch+onnxruntime 必須・無ければ skip）。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from live.inference import build_window, infer_symbol, softmax
from shared.feature_names import FEATURE_NAMES

N_FEAT = len(FEATURE_NAMES)
T0 = datetime(2026, 1, 1)


# --------------------------------------------------------------------------- #
# softmax / build_window
# --------------------------------------------------------------------------- #
def test_softmax_sums_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert p.shape == (1, 3)
    assert p.sum() == pytest.approx(1.0)
    assert np.argmax(p) == 2


def _feat_df(n, fill=1.0):
    data = {c: [float(fill)] * n for c in FEATURE_NAMES}
    data["time"] = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame(data)


def test_build_window_shape():
    df = _feat_df(20)
    w = build_window(df, 8)
    assert w.shape == (8, N_FEAT) and w.dtype == np.float32


def test_build_window_none_when_short():
    assert build_window(_feat_df(3), 8) is None


def test_build_window_none_when_nan():
    df = _feat_df(20)
    df = df.with_columns(pl.when(pl.int_range(pl.len()) == pl.len() - 1)
                         .then(float("nan")).otherwise(pl.col("ret_1")).alias("ret_1"))
    assert build_window(df, 8) is None  # 末尾窓に NaN → 推論不可


# --------------------------------------------------------------------------- #
# ONNX 予測（実エクスポート）
# --------------------------------------------------------------------------- #
def _export_tiny_model(tmp_path, seq_len=8):
    torch = pytest.importorskip("torch")
    from model.train import NormalizedModel, export_onnx
    from model.transformer import AevumTransformer, TransformerConfig

    cfg = TransformerConfig(seq_len=seq_len, d_model=16, n_heads=2, n_layers=1, dim_ff=32)
    torch.manual_seed(0)
    wrapped = NormalizedModel(AevumTransformer(cfg), np.zeros(N_FEAT), np.ones(N_FEAT), 5.0).eval()
    onnx_path = tmp_path / "model.onnx"
    export_onnx(wrapped, onnx_path, seq_len=seq_len, n_features=N_FEAT)
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps({
        "seq_len": seq_len, "feature_names": list(FEATURE_NAMES), "model_version": "test-v1",
    }), encoding="utf-8")
    return onnx_path, meta_path


def test_predictor_predict(tmp_path):
    pytest.importorskip("onnxruntime")
    from live.inference import OnnxPredictor

    onnx_path, meta_path = _export_tiny_model(tmp_path)
    pred = OnnxPredictor(onnx_path, meta_path)
    assert pred.seq_len == 8 and pred.model_version == "test-v1"
    out = pred.predict(np.random.randn(8, N_FEAT).astype(np.float32))
    assert out["probs"].shape == (3,)
    assert out["probs"].sum() == pytest.approx(1.0)
    assert out["signal"] in (-1, 0, 1)


# --------------------------------------------------------------------------- #
# infer_symbol（特徴量計算 → 予測レコード）
# --------------------------------------------------------------------------- #
def _synth(n=400, snaps_per_bar=2):
    rng = np.random.default_rng(0)
    bars, book, funding = {}, {}, {}
    times = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    for j, s in enumerate(("BTC", "ETH")):
        close = (60000 if s == "BTC" else 3000) * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        openp = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
        bars[s] = pl.DataFrame({
            "time": times, "open": openp, "high": high, "low": low, "close": close,
            "volume": np.abs(rng.normal(100, 20, n)), "trades": (np.arange(n) % 50 + 1),
        })
        # 板スナップショット（バーごとに複数）
        brows = []
        for i, t in enumerate(times):
            mid = close[i]
            for k in range(snaps_per_bar):
                brows.append({
                    "time": t + timedelta(seconds=120 * k),
                    "bid_px": [mid - 0.5 * (d + 1) for d in range(5)],
                    "bid_sz": list(np.abs(rng.normal(5, 1, 5))),
                    "ask_px": [mid + 0.5 * (d + 1) for d in range(5)],
                    "ask_sz": list(np.abs(rng.normal(5, 1, 5))),
                })
        book[s] = pl.DataFrame(brows)
        funding[s] = pl.DataFrame({
            "time": times,
            "funding_rate": rng.normal(1e-5, 1e-5, n),
            "open_interest": np.abs(rng.normal(1e6, 1e5, n)),
        })
    return bars, book, funding


def test_infer_symbol_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    from live.inference import OnnxPredictor

    onnx_path, meta_path = _export_tiny_model(tmp_path, seq_len=8)
    pred = OnnxPredictor(onnx_path, meta_path)
    bars, book, funding = _synth(n=400)
    rec = infer_symbol(pred, "BTC", bars, book, funding)
    assert rec is not None
    assert rec["symbol"] == "BTC" and rec["model_version"] == "test-v1"
    assert rec["signal"] in (-1, 0, 1)
    assert rec["prob_down"] + rec["prob_flat"] + rec["prob_up"] == pytest.approx(1.0)
    assert np.isfinite(rec["sigma"]) and rec["sigma"] > 0
    assert rec["time"] == bars["BTC"]["time"][-1]


def test_infer_symbol_none_when_insufficient(tmp_path):
    pytest.importorskip("onnxruntime")
    from live.inference import OnnxPredictor

    onnx_path, meta_path = _export_tiny_model(tmp_path, seq_len=8)
    pred = OnnxPredictor(onnx_path, meta_path)
    bars, book, funding = _synth(n=5)  # seq_len 未満
    assert infer_symbol(pred, "BTC", bars, book, funding) is None


# --------------------------------------------------------------------------- #
# _fetch_recent: 板バッファは funding と同じ WARMUP 全体（疎 board 回帰・軸A）
# --------------------------------------------------------------------------- #
def test_fetch_recent_book_spans_full_warmup_under_sparse_board(monkeypatch):
    # 回帰: 旧実装は board だけ直近 seq_len+64=192 本に打ち切っていた。疎 board 下では
    # モデル窓先頭で spread_z_60 がバルク（学習時）に対し片方向 NaN 化し train/live が乖離した。
    # 修正後は board も bar_start（funding と同じ WARMUP 全体）まで保持し、乖離が消えることを担保。
    import asyncio
    import math

    import live.inference as inf
    from data.features import WARMUP_BARS, compute_features
    from shared.feature_names import FEATURE_NAMES

    seq_len = 128
    n = WARMUP_BARS + seq_len + 400          # "DB" は live 窓より広い
    times = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    rng = np.random.default_rng(0)

    def make_bars(base):
        close = base * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        openp = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
        low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
        return pl.DataFrame({
            "time": times, "open": openp, "high": high, "low": low, "close": close,
            "volume": np.abs(rng.normal(100, 20, n)), "trades": (np.arange(n) % 50 + 1),
        })

    # 疎 board: 各バー1スナップショットだが ~45% を欠落（再接続/停止ギャップ模擬 → 60窓を飢餓させる）
    def make_book():
        rows = []
        for i, t in enumerate(times):
            if rng.random() < 0.45:
                continue
            mid = 100.0 + 5.0 * math.sin(i / 11.0)
            hs = 0.05 + 0.04 * abs(math.sin(i / 7.0))
            rows.append({
                "time": t,
                "bid_px": [mid - hs - 0.5 * d for d in range(5)],
                "bid_sz": list(np.abs(rng.normal(5, 1, 5))),
                "ask_px": [mid + hs + 0.5 * d for d in range(5)],
                "ask_sz": list(np.abs(rng.normal(5, 1, 5))),
            })
        return pl.DataFrame(rows)

    def make_funding():
        return pl.DataFrame({
            "time": times, "funding_rate": rng.normal(1e-5, 1e-5, n),
            "open_interest": np.abs(rng.normal(1e6, 1e5, n)),
        })

    bars_db = {"BTC": make_bars(60000.0), "ETH": make_bars(3000.0)}
    book_db = {"BTC": make_book(), "ETH": make_book()}
    funding_db = {"BTC": make_funding(), "ETH": make_funding()}

    async def fake_bars(conn, s):
        return bars_db[s]

    async def fake_book(conn, s):
        return book_db[s]

    async def fake_funding(conn, s):
        return funding_db[s]

    monkeypatch.setattr(inf, "fetch_bars", fake_bars)
    monkeypatch.setattr(inf, "fetch_book", fake_book)
    monkeypatch.setattr(inf, "fetch_funding", fake_funding)

    class _P:
        seq_len = 128

    # 両銘柄の live バッファを _fetch_recent で組む（run_once と同じ流れ・cross 用に両方必要）
    bl, ko, fo = {}, {}, {}
    for s in ("BTC", "ETH"):
        bl[s], ko[s], fo[s] = asyncio.run(inf._fetch_recent(None, s, _P()))

    bars_live = bl["BTC"]
    bar_start = bars_live["time"][0]
    cutoff_192 = bars_live["time"][bars_live.height - min(bars_live.height, seq_len + 64)]
    # 修正の核心: board は 192 本 cap ではなく bar_start まで保持されている
    assert ko["BTC"]["time"].min() < cutoff_192      # 旧実装なら min >= cutoff_192
    assert ko["BTC"]["time"].min() >= bar_start       # WARMUP 窓の外は持たない

    # 軸A: live バッファから計算した末尾 seq_len の spread_z_60 が、バルク（全 DB）と一致し、
    # 片方向 NaN 不整合（bulk 有限 / live NaN）が無いこと。旧 192-cap 実装ではここが壊れる。
    live_feat = compute_features(bl, ko, fo)["BTC"]
    bulk_feat = compute_features(bars_db, book_db, funding_db)["BTC"]
    live_z = live_feat.select("time", "spread_z_60").tail(seq_len).sort("time")
    bulk_z = (bulk_feat.select("time", "spread_z_60")
              .join(live_z.select("time"), on="time", how="inner").sort("time"))
    lv = live_z["spread_z_60"].to_numpy()
    bv = bulk_z["spread_z_60"].to_numpy()
    one_nan = int((np.isfinite(bv) & ~np.isfinite(lv)).sum())
    assert one_nan == 0, f"spread_z_60: {one_nan} bars finite in bulk but NaN in live (192-cap regression)"
    both = np.isfinite(bv) & np.isfinite(lv)
    assert both.sum() >= seq_len // 2           # 非空虚: 十分な有限ペアで比較している
    np.testing.assert_allclose(lv[both], bv[both], rtol=1e-6, atol=1e-7)
