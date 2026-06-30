"""data/features.py（Polars 58特徴量）の単体テスト。

  1. 出力列が [symbol, time, *FEATURE_NAMES]（60列）で過不足なし
  2. shared との整合（sigma_ewma == volatility.volatility, rsi_14 == technical.rsi）
  3. ret_1 / 時刻特徴の値
  4. 板の滞在時間加重平均・OFI バー内合計（手計算）
  5. Cross の ret_spread
  6. INSERT 文の列数 == 60 / 更新列 == 58（FEATURE_NAMES から生成）
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from data.features import (
    BAR_FEATURES_INSERT_SQL,
    WARMUP_BARS,
    compute_bar_features,
    compute_book_features,
    compute_cross_features,
    compute_features,
)
from shared import technical, volatility
from shared.feature_names import FEATURE_NAMES

UTC = timezone.utc


def _bars(n=400, seed=0, start_hour=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1.0 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(openp, close) * (1.0 - np.abs(rng.normal(0, 0.002, n)))
    vol = np.abs(rng.normal(100, 20, n))
    t0 = datetime(2026, 1, 1, start_hour, tzinfo=UTC)
    time = [t0 + timedelta(minutes=5 * i) for i in range(n)]
    return pl.DataFrame({
        "time": time, "open": openp, "high": high, "low": low,
        "close": close, "volume": vol, "trades": np.arange(n) % 50 + 1,
    })


# --------------------------------------------------------------------------- #
# 列構造
# --------------------------------------------------------------------------- #
def test_output_columns_exact():
    feats = compute_features({"BTC": _bars(), "ETH": _bars(seed=1)})
    for s, df in feats.items():
        assert df.columns == ["symbol", "time", *FEATURE_NAMES]
        assert df.width == 60
        assert df["symbol"].unique().to_list() == [s]


# --------------------------------------------------------------------------- #
# shared との整合
# --------------------------------------------------------------------------- #
def test_sigma_matches_volatility_module():
    bars = _bars()
    out = compute_bar_features(bars)
    expected = volatility.volatility(bars["close"].to_numpy())
    np.testing.assert_allclose(out["sigma_ewma"].to_numpy(), expected, equal_nan=True, rtol=1e-12)


def test_rsi_matches_technical_module():
    bars = _bars()
    out = compute_bar_features(bars)
    expected = technical.rsi(bars["close"].to_numpy(), 14)
    np.testing.assert_allclose(out["rsi_14"].to_numpy(), expected, equal_nan=True, rtol=1e-12)


def test_ret_1_is_log_return():
    bars = _bars()
    out = compute_bar_features(bars)
    close = bars["close"].to_numpy()
    exp = np.full(close.size, np.nan)
    exp[1:] = np.log(close[1:] / close[:-1])
    np.testing.assert_allclose(out["ret_1"].to_numpy(), exp, equal_nan=True, rtol=1e-12)


def test_temporal_encoding():
    # 06:00 UTC → hour_sin = sin(2π·6/24) = 1
    bars = _bars(n=2, start_hour=6)
    out = compute_bar_features(bars)
    assert out["hour_sin"][0] == pytest.approx(1.0, abs=1e-9)
    assert out["hour_cos"][0] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 板集約（滞在時間加重平均・OFI 合計）
# --------------------------------------------------------------------------- #
def _book_row(t, b0, a0, bsz0, asz0):
    # 5段の板（価格は最良から離れる方向）。OBI/OFI の検証に十分。
    bid_px = [b0 - 0.5 * i for i in range(5)]
    ask_px = [a0 + 0.5 * i for i in range(5)]
    bid_sz = [bsz0, 1.0, 1.0, 1.0, 1.0]
    ask_sz = [asz0, 1.0, 1.0, 1.0, 1.0]
    return {"time": t, "bid_px": bid_px, "bid_sz": bid_sz, "ask_px": ask_px, "ask_sz": ask_sz}


def test_book_time_weighted_obi():
    bar = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    book = pl.DataFrame([
        _book_row(bar, 100.0, 101.0, 10.0, 10.0),                       # obi_l1=0, dwell 60s
        _book_row(bar + timedelta(seconds=60), 100.0, 101.0, 30.0, 10.0),  # obi_l1=0.5, dwell 240s
    ])
    agg = compute_book_features(book)
    # 時間加重: (0*60 + 0.5*240)/300 = 0.4
    assert agg["obi_l1"][0] == pytest.approx(0.4, abs=1e-9)


def test_book_ofi_sum_sign():
    bar = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    book = pl.DataFrame([
        _book_row(bar, 100.0, 101.0, 5.0, 5.0),
        _book_row(bar + timedelta(seconds=30), 100.5, 101.0, 8.0, 4.0),
    ])
    # e_bid = 8 (bpx up), e_ask = 4 - 5 = -1 (apx flat) → ofi = 8-(-1) = 9
    agg = compute_book_features(book)
    assert agg["ofi"][0] == pytest.approx(9.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Cross
# --------------------------------------------------------------------------- #
def test_cross_ret_spread():
    feats = compute_features({"BTC": _bars(seed=0), "ETH": _bars(seed=1)})
    btc = feats["BTC"]
    eth = feats["ETH"]
    spread = btc["cross_ret_spread"].to_numpy()
    exp = btc["ret_1"].to_numpy() - eth["ret_1"].to_numpy()
    np.testing.assert_allclose(spread, exp, equal_nan=True, rtol=1e-10)


def test_cross_corr_self_is_one():
    # 同一系列同士の相関は 1。compute_cross_features を直接検証。
    bars = _bars()
    rf = compute_bar_features(bars).select("time", "ret_1")
    cross = compute_cross_features(rf, rf)
    corr = cross["cross_corr_60"].to_numpy()
    valid = corr[np.isfinite(corr)]
    assert valid.size > 0
    np.testing.assert_allclose(valid, 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# INSERT 文
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# バルク == 逐次（ローリング窓）一致：train/live 整合性の核心
# --------------------------------------------------------------------------- #
def test_bar_features_bulk_equals_incremental():
    # 長い系列のバルク計算と、末尾 WARMUP_BARS+K バーだけのローリング窓計算で、
    # 末尾 K 行が一致する（履歴依存の sigma_ewma/Wilder TA/ret_240 を含む）。
    K = 128
    full = _bars(n=2500, seed=3)
    bulk = compute_bar_features(full)
    feat_cols = [c for c in bulk.columns if c != "time"]
    tail = full.tail(WARMUP_BARS + K)
    inc = compute_bar_features(tail)
    np.testing.assert_allclose(
        inc.select(feat_cols).to_numpy()[-K:],
        bulk.select(feat_cols).to_numpy()[-K:],
        rtol=1e-6, atol=1e-7, equal_nan=True,
    )


def test_compute_features_bulk_equals_incremental():
    # 全パイプライン（2銘柄・cross 含む）でも末尾 K 行がバルクと一致。
    K = 128
    bars = {"BTC": _bars(n=2500, seed=0), "ETH": _bars(n=2500, seed=1)}
    bulk = compute_features(bars)["BTC"]
    tail_bars = {s: df.tail(WARMUP_BARS + K) for s, df in bars.items()}
    inc = compute_features(tail_bars)["BTC"]
    np.testing.assert_allclose(
        inc.select(FEATURE_NAMES).to_numpy()[-K:],
        bulk.select(FEATURE_NAMES).to_numpy()[-K:],
        rtol=1e-6, atol=1e-7, equal_nan=True,
    )


def _book_for_bars(bars, start_idx=0, snaps_per_bar=3, seed=7):
    # bars の time レンジに沿って、1バーあたり snaps_per_bar 個の板スナップショットを生成。
    # spread_z_60/spread_vol_30 が縮退しないよう、バーごとに最良気配と spread を変動させる。
    # start_idx > 0 にすると板開始を遅らせ、ライブの疎な開始（板が candle より遅く始まる）を模す。
    rng = np.random.default_rng(seed)
    times = bars["time"].to_list()
    rows = []
    for i in range(start_idx, len(times)):
        t0 = times[i]
        mid = 100.0 + 5.0 * math.sin(i / 11.0)            # バー間で中値が動く
        for j in range(snaps_per_bar):
            # バー内 snaps_per_bar 個を等間隔に配置（最後がバー終端を超えないよう 300/snaps 秒刻み）
            t = t0 + timedelta(seconds=int(j * (300 // snaps_per_bar)))
            half_spread = 0.05 + 0.04 * abs(math.sin(i / 7.0 + 0.5 * j)) + 0.01 * rng.standard_normal()
            half_spread = max(half_spread, 0.005)
            b0 = mid - half_spread
            a0 = mid + half_spread
            bsz0 = 5.0 + 4.0 * abs(rng.standard_normal())
            asz0 = 5.0 + 4.0 * abs(rng.standard_normal())
            rows.append(_book_row(t, b0, a0, bsz0, asz0))
    return pl.DataFrame(rows)


def _funding_for_bars(bars, start_idx=0, every=1, seed=9):
    # バー時刻（または every バーごと）に funding_oi 行を生成。
    # funding_z_60/oi_change が縮退しないよう rate と open_interest をバー間で変動させる。
    rng = np.random.default_rng(seed)
    times = bars["time"].to_list()
    rows = []
    oi = 1.0e7
    for i in range(start_idx, len(times)):
        if (i - start_idx) % every != 0:
            continue
        rate = 1.0e-4 * math.sin(i / 13.0) + 2.0e-5 * rng.standard_normal()
        oi *= math.exp(0.01 * rng.standard_normal())
        rows.append({"time": times[i], "funding_rate": float(rate), "open_interest": float(oi)})
    return pl.DataFrame(rows)


def test_book_funding_bulk_equals_incremental():
    # 板/funding 由来の履歴依存特徴量（spread_z_60, spread_vol_30, funding_z_60, oi_change）が
    # バルク計算とローリング窓（末尾 WARMUP_BARS+K）で末尾 K 行一致することを証明する（軸A）。
    # 既存の bulk==incremental テストは book/funding を空 dict で渡すため全 NULL になり、これらは
    # 未検証だった。ここでは実データ相当の板/funding を合成して非縮退・非空虚な一致を確認する。
    K = 128
    N = 2500
    bars = {"BTC": _bars(n=N, seed=0), "ETH": _bars(n=N, seed=1)}
    # 板はバー600から開始（ライブの疎な開始を模す）。tail 開始(=N-(WARMUP_BARS+K)=1372)より十分前なので、
    # 比較対象の末尾Kでは bulk/incremental 双方とも板が密に存在し、有限要素履歴が一致する。
    book = {
        "BTC": _book_for_bars(bars["BTC"], start_idx=600, snaps_per_bar=3, seed=7),
        "ETH": _book_for_bars(bars["ETH"], start_idx=600, snaps_per_bar=2, seed=8),
    }
    funding = {
        "BTC": _funding_for_bars(bars["BTC"], start_idx=300, every=1, seed=9),
        "ETH": _funding_for_bars(bars["ETH"], start_idx=300, every=1, seed=10),
    }

    bulk = compute_features(bars, book, funding)["BTC"]

    tail_bars = {s: df.tail(WARMUP_BARS + K) for s, df in bars.items()}
    tmin = tail_bars["BTC"]["time"].min()
    # 板/funding も tail の最小時刻以降だけを渡す（ライブのバッファ相当）。
    tail_book = {s: df.filter(pl.col("time") >= tmin) for s, df in book.items()}
    tail_funding = {s: df.filter(pl.col("time") >= tmin) for s, df in funding.items()}
    inc = compute_features(tail_bars, tail_book, tail_funding)["BTC"]

    bulk_tail = bulk.select(FEATURE_NAMES).to_numpy()[-K:]
    inc_tail = inc.select(FEATURE_NAMES).to_numpy()[-K:]

    # 非空虚性: 履歴依存の板/funding 特徴量が末尾Kで十分な有限値を持つこと（両側 NULL の空虚 PASS を防ぐ）。
    hist_feats = ["spread_z_60", "spread_vol_30", "funding_z_60", "oi_change"]
    finite_counts = {}
    for name in hist_feats:
        idx = FEATURE_NAMES.index(name)
        nb = int(np.isfinite(bulk_tail[:, idx]).sum())
        ni = int(np.isfinite(inc_tail[:, idx]).sum())
        finite_counts[name] = (nb, ni)
        assert nb >= K // 3, f"{name}: bulk finite {nb} < {K // 3} (degenerate/all-NaN)"
        assert ni >= K // 3, f"{name}: incremental finite {ni} < {K // 3} (degenerate/all-NaN)"

    print("finite_counts (bulk, inc):", finite_counts)

    np.testing.assert_allclose(
        inc_tail, bulk_tail, rtol=1e-6, atol=1e-7, equal_nan=True,
    )


def test_insert_sql_arity():
    nph = max(int(m) for m in re.findall(r"\$(\d+)", BAR_FEATURES_INSERT_SQL))
    assert nph == 2 + len(FEATURE_NAMES) == 60
    # ON CONFLICT 更新列は FEATURE_NAMES の58個（symbol/time は更新しない）
    assert BAR_FEATURES_INSERT_SQL.count("EXCLUDED.") == len(FEATURE_NAMES) == 58
