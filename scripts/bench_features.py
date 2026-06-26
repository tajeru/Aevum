"""scripts/bench_features.py — ライブ推論の特徴量計算ベンチ.

CLAUDE.md の要件B: Pi 5 上で features.py の Polars 特徴量計算が 5分バー間隔
（=300秒予算）に十分間に合うかを測る。実機 Pi でこのスクリプトを実行すること。

ライブ1回ぶんの compute_features を、現実的なバッファサイズ（bars = WARMUP_BARS +
seq_len, 2銘柄）と板スナップショット密度を変えて計測する。生成時間は計測対象外。

実行: python scripts/bench_features.py

参考ベースライン（PC: AMD Ryzen 12コア, median）。Pi 5(4コア A76)は概ね 5〜10倍遅い想定。
予算は5分バー = 300_000 ms なので、いずれも桁違いに余裕がある。
  板なし(bar+cross+funding)            ~99 ms
  板 直近200本 x 60/本   (12k snaps)   ~163 ms
  板 直近200本 x 300/本  (60k snaps)   ~405 ms
  板 全1128本 x 60/本    (68k snaps)   ~450 ms
※ live/inference._fetch_recent は板を直近 seq_len+64 本に限定するため、実運用は
  上の「直近200本」相当が上限。
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.features import WARMUP_BARS, compute_features  # noqa: E402

SEQ_LEN = 128
N_BARS = WARMUP_BARS + SEQ_LEN  # ライブの最小バッファ
T0 = datetime(2026, 1, 1)


def _bars(n, seed):
    rng = np.random.default_rng(seed)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    times = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame({
        "time": times, "open": openp, "high": high, "low": low, "close": close,
        "volume": np.abs(rng.normal(100, 20, n)), "trades": (np.arange(n) % 50 + 1),
    })


def _book(n_bars, snaps_per_bar, seed):
    rng = np.random.default_rng(seed)
    total = n_bars * snaps_per_bar
    base_t = T0 + timedelta(minutes=5 * (N_BARS - n_bars))  # 末尾 n_bars 本ぶん
    times, bpx, bsz, apx, asz = [], [], [], [], []
    for i in range(n_bars):
        mid = 60000 + i
        for k in range(snaps_per_bar):
            times.append(base_t + timedelta(minutes=5 * i, seconds=int(300 * k / snaps_per_bar)))
            bpx.append([mid - 0.5 * (d + 1) for d in range(5)])
            apx.append([mid + 0.5 * (d + 1) for d in range(5)])
            bsz.append(list(np.abs(rng.normal(5, 1, 5))))
            asz.append(list(np.abs(rng.normal(5, 1, 5))))
    return pl.DataFrame({"time": times, "bid_px": bpx, "bid_sz": bsz, "ask_px": apx, "ask_sz": asz}), total


def _funding(n, seed):
    rng = np.random.default_rng(seed)
    times = [T0 + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame({
        "time": times, "funding_rate": rng.normal(1e-5, 1e-5, n),
        "open_interest": np.abs(rng.normal(1e6, 1e5, n)),
    })


def bench(book_bars, snaps_per_bar, repeats=5):
    bars = {"BTC": _bars(N_BARS, 0), "ETH": _bars(N_BARS, 1)}
    funding = {"BTC": _funding(N_BARS, 2), "ETH": _funding(N_BARS, 3)}
    if book_bars:
        b_btc, tot = _book(book_bars, snaps_per_bar, 4)
        b_eth, _ = _book(book_bars, snaps_per_bar, 5)
        book = {"BTC": b_btc, "ETH": b_eth}
    else:
        book, tot = None, 0
    compute_features(bars, book, funding)  # warmup（JIT/キャッシュ）
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        compute_features(bars, book, funding)
        ts.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(ts)), tot


def main():
    print(f"bars/symbol = {N_BARS} (WARMUP_BARS={WARMUP_BARS} + seq_len={SEQ_LEN}), 2 symbols")
    print(f"5分バー予算 = 300_000 ms\n")
    print(f"{'scenario':<32}{'book_snaps/sym':>16}{'median_ms':>12}")
    scenarios = [
        ("板なし(bar+cross+funding)", 0, 0),
        ("板: 直近200本 x 60/本", 200, 60),
        ("板: 直近200本 x 300/本", 200, 300),
        ("板: 全1128本 x 60/本", N_BARS, 60),
    ]
    for name, bb, spb in scenarios:
        ms, tot = bench(bb, spb)
        print(f"{name:<32}{tot:>16}{ms:>12.1f}")


if __name__ == "__main__":
    main()
