"""shared/technical.py の単体テスト。

EMA/MACD/Stoch/%B は pandas を参照実装として突合（AFML/σ と同じ検証思想）。
Wilder 系（RSI/ATR/ADX）は性質＋手計算＋pandas で確認。
all_technical のキーが shared.feature_names の technical カテゴリと一致することも確認。
"""
from __future__ import annotations

import numpy as np
import pytest

from shared import technical as ta
from shared.feature_names import FEATURE_CATEGORIES


# --------------------------------------------------------------------------- #
# ema / macd（pandas adjust=False と一致）
# --------------------------------------------------------------------------- #
def test_ema_matches_pandas_adjust_false():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200).cumsum() + 100
    got = ta.ema(x, 12)
    ref = pd.Series(x).ewm(span=12, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_macd_matches_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    close = rng.normal(0, 1, 300).cumsum() + 100
    line, sig, hist = ta.macd(close)
    s = pd.Series(close)
    ref_line = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean())
    ref_sig = ref_line.ewm(span=9, adjust=False).mean()
    np.testing.assert_allclose(line, ref_line.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sig, ref_sig.to_numpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(hist, (ref_line - ref_sig).to_numpy(), rtol=1e-10, atol=1e-12)


# --------------------------------------------------------------------------- #
# wilder_rma
# --------------------------------------------------------------------------- #
def test_wilder_rma_seed_and_recursion():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ta.wilder_rma(x, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == pytest.approx(2.0)                 # mean(1,2,3)
    assert out[3] == pytest.approx((2.0 * 2 + 4.0) / 3)  # 2.6667
    assert out[4] == pytest.approx((out[3] * 2 + 5.0) / 3)


def test_wilder_rma_skips_leading_nans():
    x = np.array([np.nan, np.nan, 3.0, 6.0, 9.0, 12.0])
    out = ta.wilder_rma(x, 3)
    assert np.all(np.isnan(out[:4]))
    assert out[4] == pytest.approx(6.0)  # mean(3,6,9)


# --------------------------------------------------------------------------- #
# rsi
# --------------------------------------------------------------------------- #
def test_rsi_monotonic_and_flat():
    up = np.arange(1.0, 30.0)
    down = np.arange(30.0, 1.0, -1.0)
    flat = np.full(30, 50.0)
    r_up = ta.rsi(up, 14)
    r_down = ta.rsi(down, 14)
    r_flat = ta.rsi(flat, 14)
    assert np.nanmax(r_up) == pytest.approx(100.0)
    assert np.nanmin(r_up[np.isfinite(r_up)]) == pytest.approx(100.0)
    assert np.nanmax(r_down[np.isfinite(r_down)]) == pytest.approx(0.0)
    assert np.nanmin(r_flat[np.isfinite(r_flat)]) == pytest.approx(50.0)


def test_rsi_bounds_and_warmup():
    rng = np.random.default_rng(3)
    close = rng.normal(0, 1, 100).cumsum() + 100
    r = ta.rsi(close, 14)
    valid = r[np.isfinite(r)]
    assert np.all((valid >= 0.0) & (valid <= 100.0))
    assert np.all(np.isnan(r[:14]))   # 最初の RSI は index 14
    assert np.isfinite(r[14])


# --------------------------------------------------------------------------- #
# true_range / atr
# --------------------------------------------------------------------------- #
def test_true_range_manual():
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([9.0, 10.0, 8.0])
    close = np.array([9.5, 11.0, 9.0])
    tr = ta.true_range(high, low, close)
    assert tr[0] == pytest.approx(1.0)                       # 10-9
    assert tr[1] == pytest.approx(max(2.0, 2.5, 0.5))        # |12-9.5|=2.5
    assert tr[2] == pytest.approx(max(3.0, 0.0, 3.0))        # 11-8=3, |8-11|=3


def test_atr_normalized():
    rng = np.random.default_rng(4)
    close = rng.normal(0, 1, 50).cumsum() + 100
    high = close + 1.0
    low = close - 1.0
    a = ta.atr(high, low, close, 14, normalize=True)
    raw = ta.atr(high, low, close, 14, normalize=False)
    np.testing.assert_allclose(a, raw / close, equal_nan=True)
    assert np.isnan(a[12]) and np.isfinite(a[13])  # seed at index period-1=13


# --------------------------------------------------------------------------- #
# stoch_k / bb_pctb（pandas rolling と一致）
# --------------------------------------------------------------------------- #
def test_stoch_k_matches_rolling():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    close = rng.normal(0, 1, 80).cumsum() + 100
    high = close + np.abs(rng.normal(0, 1, 80))
    low = close - np.abs(rng.normal(0, 1, 80))
    got = ta.stoch_k(high, low, close, 14)
    hh = pd.Series(high).rolling(14).max()
    ll = pd.Series(low).rolling(14).min()
    ref = ((pd.Series(close) - ll) / (hh - ll) * 100.0).to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_bb_pctb_matches_rolling():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(6)
    close = rng.normal(0, 1, 80).cumsum() + 100
    got = ta.bb_pctb(close, 20, 2.0)
    m = pd.Series(close).rolling(20).mean()
    s = pd.Series(close).rolling(20).std(ddof=0)
    ref = ((pd.Series(close) - (m - 2 * s)) / (4 * s)).to_numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


# --------------------------------------------------------------------------- #
# adx（性質）
# --------------------------------------------------------------------------- #
def test_adx_uptrend_is_strong_and_bounded():
    n = 120
    close = np.linspace(100, 200, n)       # 強い上昇トレンド
    high = close + 0.5
    low = close - 0.5
    a = ta.adx(high, low, close, 14)
    valid = a[np.isfinite(a)]
    assert valid.size > 0
    assert np.all((valid >= 0.0) & (valid <= 100.0))
    assert np.nanmax(a) > 50.0              # トレンドで ADX 高い


def test_adx_warmup_nan():
    rng = np.random.default_rng(7)
    close = rng.normal(0, 1, 60).cumsum() + 100
    a = ta.adx(close + 1, close - 1, close, 14)
    assert np.isnan(a[0])
    assert np.all(np.isnan(a[:14]))


# --------------------------------------------------------------------------- #
# all_technical のキー == feature_names の technical カテゴリ
# --------------------------------------------------------------------------- #
def test_all_technical_keys_match_feature_names():
    rng = np.random.default_rng(8)
    close = rng.normal(0, 1, 60).cumsum() + 100
    out = ta.all_technical(close + 1, close - 1, close)
    assert tuple(out.keys()) == FEATURE_CATEGORIES["technical"]
    for v in out.values():
        assert v.shape == close.shape
