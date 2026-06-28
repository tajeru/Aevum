"""data/labels.py（Triple-Barrier）の単体テスト。

合成価格パスで正解が分かる形で検証する:
  1. cusum_filter — 無イベント / 上抜け / 下抜け / 動的閾値
  2. triple_barrier_labels — pt / sl / vertical / 同一バー両抜け / 打ち切り / NaNσ除外
  3. average_uniqueness_weights — 手計算と一致
  4. label_dataframe — 統合（行構造・ラベル整合）と INSERT プレースホルダ数一致
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from data.labels import (
    DEFAULT_HORIZON,
    LABELS_INSERT_SQL,
    average_uniqueness_weights,
    cusum_filter,
    label_dataframe,
    triple_barrier_labels,
)


# --------------------------------------------------------------------------- #
# cusum_filter
# --------------------------------------------------------------------------- #
def test_cusum_no_events_when_flat():
    assert cusum_filter(np.zeros(20), 0.05).size == 0


def test_cusum_positive_drift_scalar_threshold():
    # 全リターン +0.01、閾値 0.025 → 累積 0.03 で発火しリセット
    r = np.full(9, 0.01)
    np.testing.assert_array_equal(cusum_filter(r, 0.025), np.array([2, 5, 8]))


def test_cusum_negative_drift():
    r = np.full(9, -0.01)
    np.testing.assert_array_equal(cusum_filter(r, 0.025), np.array([2, 5, 8]))


def test_cusum_skips_nonfinite_and_bad_threshold():
    r = np.array([np.nan, 0.03, 0.03])
    # 先頭 NaN は判定スキップ。i=1 で発火→リセット、i=2 で再蓄積し再発火。
    th = np.array([np.nan, 0.02, 0.02])
    np.testing.assert_array_equal(cusum_filter(r, th), np.array([1, 2]))


# --------------------------------------------------------------------------- #
# triple_barrier_labels
# --------------------------------------------------------------------------- #
def _flat(n, val=100.0):
    return np.full(n, val, dtype=np.float64)


def _setup(n=6, sigma=0.01):
    high, low, close = _flat(n), _flat(n), _flat(n)
    sig = np.full(n, sigma, dtype=np.float64)
    return high, low, close, sig


def test_tbl_pt_touch():
    high, low, close, sig = _setup()
    # t0=1, horizon=2 → scan j=2,3。pt_level=100*exp(0.01*sqrt2)≈101.42
    high[2] = 105.0
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2)
    assert out["label"].tolist() == [1]
    assert out["touch_idx"].tolist() == [2]
    assert out["ret"][0] == pytest.approx(0.01 * math.sqrt(2))  # = w_up


def test_tbl_sl_touch():
    high, low, close, sig = _setup()
    low[3] = 95.0
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2)
    assert out["label"].tolist() == [-1]
    assert out["touch_idx"].tolist() == [3]
    assert out["ret"][0] == pytest.approx(-0.01 * math.sqrt(2))


def test_tbl_vertical():
    high, low, close, sig = _setup()
    close[3] = 100.5  # 縦バリア（t1=3）の実現リターン
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2)
    assert out["label"].tolist() == [0]
    assert out["touch_idx"].tolist() == [3]
    assert out["ret"][0] == pytest.approx(math.log(100.5 / 100.0))


def test_tbl_same_bar_both_touched_is_conservative_sl():
    high, low, close, sig = _setup()
    high[2], low[2] = 105.0, 95.0  # 同一バーで pt も sl も抜ける
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2)
    assert out["label"].tolist() == [-1]      # 保守的に sl
    assert out["touch_idx"].tolist() == [2]


def test_tbl_drops_event_without_full_horizon():
    high, low, close, sig = _setup(n=6)
    # t0=5, horizon=2 → 5+2=7 > 5 → 除外
    out = triple_barrier_labels(high, low, close, sig, [5], horizon=2)
    assert out["t0_idx"].size == 0


def test_tbl_skips_nan_sigma():
    high, low, close, sig = _setup()
    sig[1] = np.nan
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2)
    assert out["t0_idx"].size == 0


def test_tbl_barrier_levels_use_scale_to_horizon():
    # バリア幅の σ→保有期間スケールは shared.volatility.scale_to_horizon を経由すること
    # （式の二重実装禁止）。非対称倍率で式を一意に固定する。
    from shared import volatility

    high, low, close, sig = _setup(sigma=0.01)
    out = triple_barrier_labels(high, low, close, sig, [1], horizon=2,
                                pt_mult=1.3, sl_mult=0.7)
    scaled = volatility.scale_to_horizon(0.01, 2)
    assert out["pt_level"][0] == pytest.approx(100.0 * math.exp(1.3 * scaled))
    assert out["sl_level"][0] == pytest.approx(100.0 * math.exp(-0.7 * scaled))


# --------------------------------------------------------------------------- #
# average_uniqueness_weights
# --------------------------------------------------------------------------- #
def test_uniqueness_non_overlapping_is_one():
    # 区間 [0,1] と [2,3] は重複なし → 一意性 1.0
    w = average_uniqueness_weights([0, 2], [1, 3], n=4)
    np.testing.assert_allclose(w, [1.0, 1.0])


def test_uniqueness_overlapping_hand_computed():
    # event0:[0,2], event1:[1,3], n=4
    # concurrency=[1,2,2,1], inv=[1,.5,.5,1]
    # ev0 平均(0..2)=(1+.5+.5)/3, ev1 平均(1..3)=(.5+.5+1)/3
    w = average_uniqueness_weights([0, 1], [2, 3], n=4)
    np.testing.assert_allclose(w, [2.0 / 3.0, 2.0 / 3.0])


def test_uniqueness_empty():
    assert average_uniqueness_weights([], [], n=10).size == 0


def test_uniqueness_clamped_to_one_under_fp_rounding():
    # prefix-sum 差分の浮動小数点丸めで重みが 1.0 を僅かに超える決定的配置
    # （クランプ前は max≈1.0000000000000009）。一意性は定義上 <= 1。
    t0 = [6, 13, 15, 16, 26]
    tt = [16, 14, 25, 22, 27]
    w = average_uniqueness_weights(t0, tt, n=29)
    assert np.all(w <= 1.0)
    assert np.all(w > 0.0)


# --------------------------------------------------------------------------- #
# label_dataframe（統合）
# --------------------------------------------------------------------------- #
def _synthetic_series(n=800, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    # high/low は終値の周囲に微小レンジ
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.002, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.002, n)))
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = [t0 + timedelta(minutes=5 * i) for i in range(n)]
    return time, high, low, close


def test_label_dataframe_structure_and_consistency():
    time, high, low, close = _synthetic_series()
    rows = label_dataframe("BTC", time, high, low, close,
                           horizon=DEFAULT_HORIZON, sigma_span=50)
    assert len(rows) > 0
    name = {1: "pt", -1: "sl", 0: "vertical"}
    for r in rows:
        assert len(r) == 14
        symbol, t, label, ret, sigma, pt_lvl, sl_lvl, ptm, slm, hb, t1, tt, barrier, wgt = r
        assert symbol == "BTC"
        assert label in (-1, 0, 1)
        assert barrier == name[label]
        assert sigma > 0.0
        assert sl_lvl < pt_lvl                     # 下側 < 上側バリア
        assert isinstance(t, datetime) and isinstance(t1, datetime) and isinstance(tt, datetime)
        assert t <= tt <= t1                        # 到達は t0..t1 の範囲
        assert 0.0 < wgt <= 1.0                     # 平均一意性
        assert hb == DEFAULT_HORIZON


def test_label_dataframe_length_mismatch_raises():
    with pytest.raises(ValueError):
        label_dataframe("BTC", [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                        [1.0, 2.0], [1.0, 2.0], [1.0, 2.0])


def test_labels_insert_arity():
    placeholders = max(int(m) for m in re.findall(r"\$(\d+)", LABELS_INSERT_SQL))
    assert placeholders == 14
    time, high, low, close = _synthetic_series(n=400)
    rows = label_dataframe("ETH", time, high, low, close, sigma_span=50)
    assert rows and len(rows[0]) == placeholders
