"""data/labels.py — PC側: Triple-Barrier ラベリング.

López de Prado『Advances in Financial Machine Learning』第3章の Triple-Barrier 法。
σ は shared/volatility.py の唯一定義を使用（per-bar σ）。バリア幅・CUSUM 閾値とも
`shared.volatility.scale_to_horizon` で保有期間にスケールし、スケール式も統一する。

確定仕様（baseline）
-------------------
* 縦バリア horizon = 48 バー（5分足 → 4時間）
* バリア幅（対数リターン空間）: w = mult × σ × √horizon
    - pt_level = close[t0] · exp(+pt_mult·σ·√horizon)
    - sl_level = close[t0] · exp(−sl_mult·σ·√horizon)
* 倍率は対称 [1.0, 1.0]（pt_mult / sl_mult は設定変更可。後で非対称も試せる）
* 到達判定はバー内 high/low（成行損切り・指値利確の実約定セマンティクスと一致）
* ラベル: pt先着=+1 / sl先着=−1 / 縦バリア到達=0
* 同一バーで pt・sl 両抜け → 保守的に sl 先着（−1）扱い（楽観バイアス回避）
* イベント抽出: 対称 CUSUM フィルタ。閾値 h = h_mult × σ × √horizon
* サンプル重み: AFML の平均一意性（concurrency ベース）を labels.sample_weight に保存
* full horizon を観測できないイベント（t0+horizon > 末尾）は落とす（打ち切り回避）

解析コア（numpy）は DB I/O から分離し、合成パスで単体テストする。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional, Sequence

import numpy as np

from shared import volatility

log = logging.getLogger("aevum.labels")

# --------------------------------------------------------------------------- #
# 既定パラメータ
# --------------------------------------------------------------------------- #
DEFAULT_HORIZON: int = 48          # 5分足 × 48 = 4時間
DEFAULT_PT_MULT: float = 1.0
DEFAULT_SL_MULT: float = 1.0
DEFAULT_H_MULT: float = 1.0        # CUSUM 閾値倍率（大きいほどイベント減）
DEFAULT_SIGMA_SPAN: int = volatility.DEFAULT_SPAN

# labels テーブルへの冪等 INSERT（14列）。
LABELS_INSERT_SQL: str = (
    "INSERT INTO labels "
    "(symbol, time, label, ret, sigma, pt_level, sl_level, pt_mult, sl_mult, "
    " horizon_bars, t1, touch_time, touch_barrier, sample_weight) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
    "ON CONFLICT (symbol, time) DO UPDATE SET "
    "label = EXCLUDED.label, ret = EXCLUDED.ret, sigma = EXCLUDED.sigma, "
    "pt_level = EXCLUDED.pt_level, sl_level = EXCLUDED.sl_level, "
    "pt_mult = EXCLUDED.pt_mult, sl_mult = EXCLUDED.sl_mult, "
    "horizon_bars = EXCLUDED.horizon_bars, t1 = EXCLUDED.t1, "
    "touch_time = EXCLUDED.touch_time, touch_barrier = EXCLUDED.touch_barrier, "
    "sample_weight = EXCLUDED.sample_weight"
)

_BARRIER_NAME = {1: "pt", -1: "sl", 0: "vertical"}


# --------------------------------------------------------------------------- #
# CUSUM フィルタ（AFML Snippet 2.4）
# --------------------------------------------------------------------------- #
def cusum_filter(returns: Any, threshold: Any) -> np.ndarray:
    """対称 CUSUM フィルタ。累積上昇/下降が閾値を超えた点の index を返す。

    `returns` は（対数）リターン列。`threshold` はスカラ、または `returns` と同長の
    配列（σ ベースの動的閾値）。非有限/0以下の閾値、非有限のリターンは判定をスキップ。
    """
    r = np.asarray(returns, dtype=np.float64)
    n = r.size
    scalar = np.ndim(threshold) == 0
    th = None if scalar else np.asarray(threshold, dtype=np.float64)

    events: list[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for i in range(n):
        d = r[i]
        if not math.isfinite(d):
            continue
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        hi = float(threshold) if scalar else th[i]
        if not math.isfinite(hi) or hi <= 0.0:
            continue
        if s_neg < -hi:
            s_neg = 0.0
            events.append(i)
        elif s_pos > hi:
            s_pos = 0.0
            events.append(i)
    return np.array(events, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Triple-Barrier コア
# --------------------------------------------------------------------------- #
def triple_barrier_labels(
    high: Any,
    low: Any,
    close: Any,
    sigma: Any,
    events: Any,
    *,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, np.ndarray]:
    """各イベント起点に Triple-Barrier ラベルを付与する。

    返り値は採用イベントぶんの並列 numpy 配列の dict:
      t0_idx, label, ret, sigma, pt_level, sl_level, t1_idx, touch_idx
    σ が NaN/0以下のイベント、full horizon を観測できないイベントは除外する。
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    horizon = int(horizon)
    n = close.size

    t0s: list[int] = []
    labels: list[int] = []
    rets: list[float] = []
    sigs: list[float] = []
    pts: list[float] = []
    sls: list[float] = []
    t1s: list[int] = []
    touches: list[int] = []

    for t0 in np.asarray(events, dtype=np.int64):
        t0 = int(t0)
        if t0 + horizon > n - 1:          # full horizon が観測できない → 除外
            continue
        s = sigma[t0]
        if not math.isfinite(s) or s <= 0.0:
            continue

        scaled = volatility.scale_to_horizon(s, horizon)  # σ→保有期間スケールは唯一定義を使う
        w_up = pt_mult * scaled
        w_dn = sl_mult * scaled
        pt_level = close[t0] * math.exp(w_up)
        sl_level = close[t0] * math.exp(-w_dn)
        t1_idx = t0 + horizon

        label = 0
        touch_idx = t1_idx
        ret = math.log(close[t1_idx] / close[t0])  # 縦バリアの実現リターン
        for j in range(t0 + 1, t1_idx + 1):
            hit_pt = high[j] >= pt_level
            hit_sl = low[j] <= sl_level
            if hit_pt and hit_sl:         # 同一バー両抜け → 保守的に sl 先着
                label, touch_idx, ret = -1, j, -w_dn
                break
            if hit_pt:
                label, touch_idx, ret = 1, j, w_up
                break
            if hit_sl:
                label, touch_idx, ret = -1, j, -w_dn
                break

        t0s.append(t0)
        labels.append(label)
        rets.append(ret)
        sigs.append(s)
        pts.append(pt_level)
        sls.append(sl_level)
        t1s.append(t1_idx)
        touches.append(touch_idx)

    return {
        "t0_idx": np.array(t0s, dtype=np.int64),
        "label": np.array(labels, dtype=np.int8),
        "ret": np.array(rets, dtype=np.float64),
        "sigma": np.array(sigs, dtype=np.float64),
        "pt_level": np.array(pts, dtype=np.float64),
        "sl_level": np.array(sls, dtype=np.float64),
        "t1_idx": np.array(t1s, dtype=np.int64),
        "touch_idx": np.array(touches, dtype=np.int64),
    }


# --------------------------------------------------------------------------- #
# サンプル重み（AFML 平均一意性）
# --------------------------------------------------------------------------- #
def average_uniqueness_weights(t0_idx: Any, touch_idx: Any, n: int) -> np.ndarray:
    """各ラベルの平均一意性を返す（AFML 第4章）。

    各バーの concurrency（同時にアクティブなラベル数）の逆数を、ラベル区間
    [t0, touch] で平均したもの。重複の多いラベルほど小さくなる。
    """
    t0 = np.asarray(t0_idx, dtype=np.int64)
    tt = np.asarray(touch_idx, dtype=np.int64)
    m = t0.size
    if m == 0:
        return np.zeros(0, dtype=np.float64)

    # concurrency を差分配列で構築
    conc = np.zeros(n + 1, dtype=np.float64)
    for i in range(m):
        conc[t0[i]] += 1.0
        conc[tt[i] + 1] -= 1.0
    conc = np.cumsum(conc)[:n]

    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1.0), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])  # 長さ n+1

    w = np.empty(m, dtype=np.float64)
    for i in range(m):
        a, b = int(t0[i]), int(tt[i])
        w[i] = (prefix[b + 1] - prefix[a]) / (b - a + 1)
    # 平均一意性は定義上 (0, 1]。prefix-sum 差分の浮動小数点丸めで 1+ε(≈1e-15) に
    # なり得るため 1.0 にクランプ（数学的に厳密。下限は concurrency>=1 で保証）。
    return np.minimum(w, 1.0)


# --------------------------------------------------------------------------- #
# 統合: OHLC → labels 行（純粋・テスト可能）
# --------------------------------------------------------------------------- #
def label_dataframe(
    symbol: str,
    time: Sequence,
    high: Any,
    low: Any,
    close: Any,
    *,
    horizon: int = DEFAULT_HORIZON,
    pt_mult: float = DEFAULT_PT_MULT,
    sl_mult: float = DEFAULT_SL_MULT,
    h_mult: float = DEFAULT_H_MULT,
    sigma_span: int = DEFAULT_SIGMA_SPAN,
) -> list[tuple]:
    """1銘柄の OHLC 系列から labels テーブル行（14列タプル）のリストを作る。

    σ・log_returns・scale_to_horizon は shared/volatility.py を使用（式統一）。
    CUSUM 閾値 h = h_mult × σ × √horizon。返り値は LABELS_INSERT_SQL に対応。
    """
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = close.size
    if not (len(time) == n == high.size == low.size):
        raise ValueError("time / high / low / close は同じ長さであること")

    sigma = volatility.volatility(close, sigma_span)
    logret = volatility.log_returns(close)
    threshold = h_mult * volatility.scale_to_horizon(sigma, horizon)  # σ 参照の動的閾値
    events = cusum_filter(logret, threshold)
    core = triple_barrier_labels(
        high, low, close, sigma, events,
        pt_mult=pt_mult, sl_mult=sl_mult, horizon=horizon,
    )
    weights = average_uniqueness_weights(core["t0_idx"], core["touch_idx"], n)

    rows: list[tuple] = []
    for k in range(core["t0_idx"].size):
        t0 = int(core["t0_idx"][k])
        lbl = int(core["label"][k])
        rows.append((
            symbol,
            time[t0],
            lbl,
            float(core["ret"][k]),
            float(core["sigma"][k]),
            float(core["pt_level"][k]),
            float(core["sl_level"][k]),
            float(pt_mult),
            float(sl_mult),
            int(horizon),
            time[int(core["t1_idx"][k])],
            time[int(core["touch_idx"][k])],
            _BARRIER_NAME[lbl],
            float(weights[k]),
        ))
    return rows


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
async def fetch_ohlcv(conn: Any, symbol: str):
    """ohlcv_bars から time/high/low/close を時系列順に取得。"""
    recs = await conn.fetch(
        "SELECT time, high, low, close FROM ohlcv_bars WHERE symbol = $1 ORDER BY time",
        symbol,
    )
    time = [r["time"] for r in recs]
    high = np.array([r["high"] for r in recs], dtype=np.float64)
    low = np.array([r["low"] for r in recs], dtype=np.float64)
    close = np.array([r["close"] for r in recs], dtype=np.float64)
    return time, high, low, close


async def store_labels(conn: Any, rows: list[tuple]) -> None:
    if rows:
        await conn.executemany(LABELS_INSERT_SQL, rows)


async def run(symbols: Sequence[str] = ("BTC", "ETH"), **params) -> None:
    """各銘柄の OHLC を読み、Triple-Barrier ラベルを計算して labels へ書き込む。"""
    import asyncpg  # 遅延 import

    from data.ingestion import resolve_dsn

    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            for symbol in symbols:
                time, high, low, close = await fetch_ohlcv(conn, symbol)
                rows = label_dataframe(symbol, time, high, low, close, **params)
                await store_labels(conn, rows)
                log.info("labels stored: symbol=%s rows=%d (bars=%d)", symbol, len(rows), len(time))
    finally:
        await pool.close()


def main() -> None:
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run())


if __name__ == "__main__":
    main()
