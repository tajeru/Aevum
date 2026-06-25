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
def test_insert_sql_arity():
    nph = max(int(m) for m in re.findall(r"\$(\d+)", BAR_FEATURES_INSERT_SQL))
    assert nph == 2 + len(FEATURE_NAMES) == 60
    # ON CONFLICT 更新列は FEATURE_NAMES の58個（symbol/time は更新しない）
    assert BAR_FEATURES_INSERT_SQL.count("EXCLUDED.") == len(FEATURE_NAMES) == 58
