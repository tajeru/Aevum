"""shared/volatility.py の単体テスト。

σ は Aevum の最重要不変条件（唯一の定義）。ここでは
  1. 仕様（対数リターン / EWMA / per-bar）の固定
  2. pandas（= AFML 参照実装）との一致
  3. 端点・異常入力・不変性（非破壊・決定性）
  4. ライブ⇄バッチ整合（recommended_min_history）
を網羅する。
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.volatility import (
    DEFAULT_MIN_PERIODS,
    ewma_std,
    latest,
    log_returns,
    recommended_min_history,
    scale_to_horizon,
    volatility,
)


# --------------------------------------------------------------------------- #
# log_returns
# --------------------------------------------------------------------------- #
def test_log_returns_basic():
    close = [1.0, np.e, np.e**2]
    r = log_returns(close)
    assert np.isnan(r[0])
    np.testing.assert_allclose(r[1:], [1.0, 1.0])


def test_log_returns_is_log_not_simple():
    # log と simple を区別できる系列で、対数式であることを固定する。
    close = [1.0, 2.0, 3.0]
    r = log_returns(close)
    np.testing.assert_allclose(r[1:], [np.log(2.0), np.log(1.5)])


def test_log_returns_rejects_nonpositive():
    with pytest.raises(ValueError):
        log_returns([1.0, -1.0, 2.0])
    with pytest.raises(ValueError):
        log_returns([1.0, 0.0])


def test_log_returns_rejects_nonfinite():
    with pytest.raises(ValueError):
        log_returns([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        log_returns([1.0, np.inf])


def test_log_returns_empty_and_single():
    assert log_returns([]).shape == (0,)
    s = log_returns([100.0])
    assert s.shape == (1,) and np.isnan(s[0])


# --------------------------------------------------------------------------- #
# volatility — 形・端点
# --------------------------------------------------------------------------- #
def test_volatility_length_and_leading_nans():
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    s = volatility(close, span=5, min_periods=2)
    assert s.shape == close.shape
    # σ[0]=リターン無し, σ[1]=リターン1本(min_periods未満) → ともに NaN
    assert np.isnan(s[0])
    assert np.isnan(s[1])
    assert np.all(np.isfinite(s[2:]))


def test_volatility_constant_prices_zero():
    close = np.full(50, 123.45)
    s = volatility(close, span=10, min_periods=2)
    assert np.isnan(s[0]) and np.isnan(s[1])
    np.testing.assert_allclose(s[2:], 0.0, atol=1e-12)


def test_volatility_empty_and_single():
    assert volatility([]).shape == (0,)
    s = volatility([100.0])
    assert s.shape == (1,) and np.isnan(s[0])


def test_volatility_2d_raises():
    with pytest.raises(ValueError):
        volatility(np.ones((3, 3)))


def test_volatility_span_and_min_periods_validation():
    with pytest.raises(ValueError):
        volatility([1.0, 2.0, 3.0], span=0)
    with pytest.raises(ValueError):
        ewma_std([1.0, 2.0, 3.0], span=0)
    with pytest.raises(ValueError):
        ewma_std([1.0, 2.0, 3.0], span=5, min_periods=0)


def test_volatility_no_mutation_and_deterministic():
    close = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0])
    snapshot = close.copy()
    a = volatility(close, span=5)
    b = volatility(close, span=5)
    np.testing.assert_array_equal(a, b)              # 決定性
    np.testing.assert_array_equal(close, snapshot)   # 入力非破壊


def test_volatility_accepts_list_like():
    close = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
    np.testing.assert_allclose(
        volatility(close, span=5),
        volatility(np.array(close), span=5),
        equal_nan=True,
    )


# --------------------------------------------------------------------------- #
# pandas（= AFML 参照）との一致
# --------------------------------------------------------------------------- #
def test_ewma_std_matches_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 300)
    for bias in (False, True):
        got = ewma_std(x, span=50, min_periods=2, bias=bias)
        ref = pd.Series(x).ewm(span=50, adjust=True, min_periods=2).std(bias=bias)
        np.testing.assert_allclose(got, ref.to_numpy(), rtol=1e-8, atol=1e-12, equal_nan=True)


def test_volatility_matches_pandas_reference():
    # AFML getDailyVol と等価: log リターン → ewm(span).std()
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 500)))
    span, mp = 100, 2
    got = volatility(close, span, min_periods=mp)
    r = pd.Series(np.log(close)).diff()  # 先頭 NaN を含む（AFML 同様）
    ref = r.ewm(span=span, adjust=True, min_periods=mp, ignore_na=False).std()
    np.testing.assert_allclose(got, ref.to_numpy(), rtol=1e-7, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# latest
# --------------------------------------------------------------------------- #
def test_latest_equals_last_element():
    close = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0])
    full = volatility(close, span=5)
    assert latest(close, span=5) == pytest.approx(full[-1])


def test_latest_nan_when_undetermined():
    assert np.isnan(latest([]))
    assert np.isnan(latest([100.0]))


# --------------------------------------------------------------------------- #
# recommended_min_history / ライブ⇄バッチ整合
# --------------------------------------------------------------------------- #
def test_recommended_min_history_decay():
    span, tol = 20, 1e-8
    k = recommended_min_history(span, tol)
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    assert decay ** (k - 1) <= tol * (1.0 + 1e-9)
    assert k >= DEFAULT_MIN_PERIODS + 1


def test_recommended_min_history_validation():
    with pytest.raises(ValueError):
        recommended_min_history(0)
    with pytest.raises(ValueError):
        recommended_min_history(20, tol=0.0)
    with pytest.raises(ValueError):
        recommended_min_history(20, tol=1.0)


def test_live_buffer_matches_batch():
    span = 20
    rng = np.random.default_rng(123)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 4000)))
    full = volatility(close, span)[-1]
    k = recommended_min_history(span, tol=1e-10)
    windowed = latest(close[-k:], span)
    assert windowed == pytest.approx(full, rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- #
# scale_to_horizon
# --------------------------------------------------------------------------- #
def test_scale_to_horizon_scalar_and_array():
    assert scale_to_horizon(2.0, 4) == pytest.approx(4.0)
    assert isinstance(scale_to_horizon(1.0, 4), float)
    np.testing.assert_allclose(scale_to_horizon(np.array([1.0, 2.0]), 9), [3.0, 6.0])


def test_scale_to_horizon_rejects_nonpositive():
    with pytest.raises(ValueError):
        scale_to_horizon(1.0, 0)
    with pytest.raises(ValueError):
        scale_to_horizon(1.0, -1)
