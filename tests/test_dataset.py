"""model/dataset.py の単体テスト。

torch 非依存のコア（正規化 / walk-forward / サンプル構築）を網羅し、
SequenceDataset/make_folds は torch があれば検証（無ければ skip）。
"""
from __future__ import annotations

import numpy as np
import pytest

from model.dataset import (
    CLIP_SIGMA,
    ZScoreClipNormalizer,
    build_samples,
    make_folds,
    walk_forward_splits,
)
from shared.feature_names import N_FEATURES

FIVE_MIN = np.timedelta64(5, "m")
T0 = np.datetime64("2026-01-01T00:00")


def _times(n, start=T0):
    return start + np.arange(n) * FIVE_MIN


# --------------------------------------------------------------------------- #
# ZScoreClipNormalizer
# --------------------------------------------------------------------------- #
def test_normalizer_standardizes():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, (1000, 4))
    norm = ZScoreClipNormalizer(clip=None)
    Z = norm.fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_normalizer_clips_outliers():
    X = np.zeros((100, 1))
    X[:, 0] = np.arange(100)
    X[0, 0] = 1e6  # 外れ値
    Z = ZScoreClipNormalizer(clip=5.0).fit_transform(X)
    assert Z.max() <= 5.0 + 1e-9
    assert Z.min() >= -5.0 - 1e-9


def test_normalizer_constant_column():
    X = np.column_stack([np.full(50, 7.0), np.arange(50.0)])
    Z = ZScoreClipNormalizer(clip=5.0).fit_transform(X)
    np.testing.assert_allclose(Z[:, 0], 0.0, atol=1e-12)  # 定数列 → 0


def test_normalizer_transform_uses_fitted_stats():
    norm = ZScoreClipNormalizer(clip=None).fit(np.array([[0.0], [10.0]]))
    # mean=5, std=5 → 新データ 15 は (15-5)/5 = 2
    assert norm.transform(np.array([[15.0]]))[0, 0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# build_samples
# --------------------------------------------------------------------------- #
def test_build_samples_filters_and_maps():
    n = 20
    F = np.ones((n, N_FEATURES))
    F[:5, :] = np.nan  # ウォームアップ
    times = _times(n)
    events = {
        "e_idx": np.array([3, 10, 15]),     # e=3 は窓に NaN → 除外
        "t1": times[[3, 10, 15]] + 48 * FIVE_MIN,
        "label": np.array([-1, 0, 1]),
        "weight": np.array([0.5, 0.6, 0.7]),
    }
    s = build_samples({"BTC": F}, {"BTC": times}, {"BTC": events}, seq_len=4)
    assert s["e_idx"].tolist() == [10, 15]
    assert s["y"].tolist() == [1, 2]          # {0→1, 1→2}
    assert s["w"].tolist() == [0.6, 0.7]
    assert s["t0"].tolist() == times[[10, 15]].astype("datetime64[ns]").tolist()
    assert list(s["sym"]) == ["BTC", "BTC"]


def test_build_samples_drops_short_window():
    F = np.ones((10, N_FEATURES))
    times = _times(10)
    events = {"e_idx": np.array([2]), "t1": times[[2]], "label": np.array([1]), "weight": np.array([1.0])}
    s = build_samples({"BTC": F}, {"BTC": times}, {"BTC": events}, seq_len=8)
    assert s["e_idx"].size == 0  # e=2 < seq_len-1=7


# --------------------------------------------------------------------------- #
# walk_forward_splits（Purge + Embargo + 拡張窓）
# --------------------------------------------------------------------------- #
def test_walk_forward_purge_embargo_and_expanding():
    n = 200
    t0 = _times(n).astype("datetime64[ns]")
    t1 = t0 + 48 * FIVE_MIN
    embargo_bars = 48
    splits = walk_forward_splits(t0, t1, n_splits=3, embargo_bars=embargo_bars)
    assert len(splits) >= 2
    embargo = np.timedelta64(embargo_bars * 5, "m")

    prev_train = None
    for train_idx, test_idx in splits:
        test_start = t0[test_idx].min()
        # Purge + Embargo: train の t1 はすべて test_start - embargo より前
        assert t1[train_idx].max() < test_start - embargo
        # train と test は時間的に重ならない
        assert t0[train_idx].max() < test_start
        assert len(set(train_idx) & set(test_idx)) == 0
        # 拡張窓: train は前 fold を包含
        if prev_train is not None:
            assert set(prev_train).issubset(set(train_idx))
        prev_train = train_idx


def test_walk_forward_empty():
    assert walk_forward_splits(np.array([], dtype="datetime64[ns]"),
                               np.array([], dtype="datetime64[ns]")) == []


# --------------------------------------------------------------------------- #
# make_folds / SequenceDataset（torch）
# --------------------------------------------------------------------------- #
def _synthetic_two_symbols(n=400, seq_len=16):
    rng = np.random.default_rng(0)
    feats, times, events = {}, {}, {}
    for j, s in enumerate(("BTC", "ETH")):
        F = rng.normal(0, 1, (n, N_FEATURES))
        t = _times(n)
        e = np.arange(seq_len, n - 50, 4)
        events[s] = {
            "e_idx": e,
            "t1": t[e] + 48 * FIVE_MIN,
            "label": rng.integers(-1, 2, e.size),
            "weight": np.full(e.size, 1.0),
        }
        feats[s], times[s] = F, t
    return feats, times, events, seq_len


def test_make_folds_and_dataset():
    torch = pytest.importorskip("torch")
    feats, times, events, seq_len = _synthetic_two_symbols()
    folds = make_folds(feats, times, events, seq_len=seq_len, n_splits=2, embargo_bars=16)
    assert len(folds) >= 1
    fold = folds[0]
    assert fold.normalizer.mean_ is not None
    assert len(fold.train) > 0 and len(fold.test) > 0
    # サンプル形状
    X, y, w = fold.train[0]
    assert tuple(X.shape) == (seq_len, N_FEATURES)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long and 0 <= int(y) <= 2
    assert w.dtype == torch.float32
    # DataLoader でバッチ化できる
    loader = torch.utils.data.DataLoader(fold.train, batch_size=8)
    xb, yb, wb = next(iter(loader))
    assert xb.shape[1:] == (seq_len, N_FEATURES)
    assert xb.shape[0] == yb.shape[0] == wb.shape[0]


def test_make_folds_no_index_leakage():
    pytest.importorskip("torch")
    feats, times, events, seq_len = _synthetic_two_symbols()
    folds = make_folds(feats, times, events, seq_len=seq_len, n_splits=2, embargo_bars=16)
    for fold in folds:
        assert len(set(fold.train_idx) & set(fold.test_idx)) == 0


def test_make_folds_normalizer_fits_train_period_only():
    # 正規化リーク防止: 各 fold の正規化統計は「test_start - embargo より前のバー」だけで
    # fit され、test 期間バーは混入しないこと。時間ドリフトを入れ、test を含めると統計が
    # 変わる状況で「train 期間のみ再計算」と一致・「全期間 fit」と不一致を確認する。
    n, seq_len, embargo_bars = 400, 16, 16
    rng = np.random.default_rng(1)
    feats, times, events = {}, {}, {}
    for s in ("BTC", "ETH"):
        F = rng.normal(0.0, 1.0, (n, N_FEATURES)) + np.linspace(0.0, 10.0, n)[:, None]  # 時間ドリフト
        t = _times(n)
        e = np.arange(seq_len, n - 50, 4)
        events[s] = {
            "e_idx": e, "t1": t[e] + 48 * FIVE_MIN,
            "label": rng.integers(-1, 2, e.size), "weight": np.full(e.size, 1.0),
        }
        feats[s], times[s] = F, t

    samples = build_samples(feats, times, events, seq_len=seq_len)
    folds = make_folds(feats, times, events, seq_len=seq_len, n_splits=2, embargo_bars=embargo_bars)
    assert len(folds) >= 1
    embargo = np.timedelta64(embargo_bars * 5, "m")

    for fold in folds:
        cutoff = samples["t0"][fold.test_idx].min() - embargo
        rows = []
        for s, F in feats.items():
            tt = np.asarray(times[s], dtype="datetime64[ns]")
            r = np.asarray(F, dtype=np.float64)[tt < cutoff]
            rows.append(r[np.isfinite(r).all(axis=1)])
        indep = ZScoreClipNormalizer().fit(np.concatenate(rows, axis=0))
        # train 期間のみで再計算した統計と一致
        np.testing.assert_allclose(fold.normalizer.mean_, indep.mean_)
        np.testing.assert_allclose(fold.normalizer.std_, indep.std_)
        # 全期間 fit とは異なる = test 期間バーが fit に混入していない証拠
        full = ZScoreClipNormalizer().fit(
            np.concatenate([np.asarray(F, dtype=np.float64) for F in feats.values()], axis=0))
        assert not np.allclose(fold.normalizer.mean_, full.mean_)
