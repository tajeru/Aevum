"""model/dataset.py — PyTorch Dataset（正規化・系列窓化・Walk-forward 分割）.

CLAUDE.md の不変条件:
* 正規化はここだけ（DB は生）。統計は学習 fold のみで fit し val/test に適用（リーク防止）。
* 検証は Walk-forward + Purging + Embargo のみ（k-fold 禁止）。
  Purge 長 = 最大ラベル保有期間 = horizon。Embargo は test 直前に追加の隙間。

確定仕様
--------
* 系列長 SEQ_LEN = 128 バー（5分足）。サンプル = ラベル付きイベントのみ。
  入力 = イベント末尾までの SEQ_LEN×58 特徴量、ターゲット = label。
* ラベル写像 {-1,0,1} → {0,1,2}（CrossEntropy 用）。sample_weight も返す。
* 正規化 = z-score + ±5σ クリップ。統計は train 期間のバーのみで算出。
* Walk-forward = 拡張窓（expanding）。BTC/ETH をプール（cross 特徴があるため1モデル）。
* ウォームアップ NaN を含む窓のイベントは除外（ゼロ詰めしない）。

コア（ZScoreClipNormalizer / walk_forward_splits / build_samples）は numpy のみで
torch 非依存にテストできる。SequenceDataset は薄いラッパで torch を遅延 import。
"""
from __future__ import annotations

from collections import namedtuple
from typing import Optional

import numpy as np

from shared.feature_names import FEATURE_NAMES, N_FEATURES

SEQ_LEN: int = 128
N_SPLITS: int = 5
HORIZON: int = 48            # = labels の horizon
PURGE: int = HORIZON         # Purge 長 = 最大ラベル保有期間
EMBARGO_BARS: int = HORIZON  # 既定 Embargo（設定可）
CLIP_SIGMA: float = 5.0
BAR_MINUTES: int = 5

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


# --------------------------------------------------------------------------- #
# 正規化（z-score + クリップ）
# --------------------------------------------------------------------------- #
class ZScoreClipNormalizer:
    """各特徴を (x-mean)/std にし ±clip でクリップ。mean/std は学習データのみで fit。"""

    def __init__(self, clip: Optional[float] = CLIP_SIGMA) -> None:
        self.clip = clip
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X) -> "ZScoreClipNormalizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)  # 定数特徴は std=1（実質0化）
        return self

    def transform(self, X) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit() を先に呼ぶこと")
        X = np.asarray(X, dtype=np.float64)
        z = (X - self.mean_) / self.std_
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# サンプル構築（ラベル付きイベント → 系列窓）
# --------------------------------------------------------------------------- #
def build_samples(
    features_by_sym: dict[str, np.ndarray],
    times_by_sym: dict[str, np.ndarray],
    events_by_sym: dict[str, dict],
    *,
    seq_len: int = SEQ_LEN,
) -> dict[str, np.ndarray]:
    """ラベル付きイベントを系列サンプルへ。NaN を含む窓・系列長未満のイベントは除外。

    events_by_sym[sym] = {"e_idx": int[], "t1": datetime64[], "label": int[], "weight": float[]}
    返り値（全銘柄プール）: dict of arrays
      sym(str), e_idx(int), t0(datetime64), t1(datetime64), y(int class), w(float)
    """
    sym_list, e_list, t0_list, t1_list, y_list, w_list = [], [], [], [], [], []
    for sym, F in features_by_sym.items():
        F = np.asarray(F, dtype=np.float64)
        if F.shape[1] != N_FEATURES:
            raise ValueError(f"{sym}: 特徴量列数 {F.shape[1]} != {N_FEATURES}")
        times = np.asarray(times_by_sym[sym])
        ev = events_by_sym[sym]
        e_idx = np.asarray(ev["e_idx"], dtype=np.int64)
        t1 = np.asarray(ev["t1"])
        label = np.asarray(ev["label"], dtype=np.int64)
        weight = np.asarray(ev["weight"], dtype=np.float64)
        for k in range(e_idx.size):
            e = int(e_idx[k])
            if e < seq_len - 1:                      # 系列長に満たない
                continue
            win = F[e - seq_len + 1: e + 1]
            if not np.isfinite(win).all():           # ウォームアップ NaN を含む
                continue
            sym_list.append(sym)
            e_list.append(e)
            t0_list.append(times[e])
            t1_list.append(t1[k])
            y_list.append(LABEL_TO_CLASS[int(label[k])])
            w_list.append(float(weight[k]))

    return {
        "sym": np.array(sym_list, dtype=object),
        "e_idx": np.array(e_list, dtype=np.int64),
        "t0": np.array(t0_list, dtype="datetime64[ns]"),
        "t1": np.array(t1_list, dtype="datetime64[ns]"),
        "y": np.array(y_list, dtype=np.int64),
        "w": np.array(w_list, dtype=np.float64),
    }


# --------------------------------------------------------------------------- #
# Walk-forward 分割（Purge + Embargo）
# --------------------------------------------------------------------------- #
def walk_forward_splits(
    t0: np.ndarray,
    t1: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    embargo_bars: int = EMBARGO_BARS,
    bar_minutes: int = BAR_MINUTES,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """拡張窓 Walk-forward。各 fold: test=直後ブロック、train=それ以前（Purge+Embargo 後）。

    train は「t0 < test_start かつ t1 < test_start - embargo」を満たすサンプルのみ。
    t1 はラベル終了時刻なので、この条件で Purge（test と重なるラベルの除外）と
    Embargo（test 直前の隙間）を同時に満たす。返り値は (train_idx, test_idx) のリスト。
    """
    t0 = np.asarray(t0, dtype="datetime64[ns]")
    t1 = np.asarray(t1, dtype="datetime64[ns]")
    n = t0.size
    if n == 0:
        return []
    ts = np.sort(t0)
    qs = np.linspace(0.0, 1.0, n_splits + 2)
    edges = [ts[min(int(round(q * n)), n - 1)] for q in qs]
    embargo = np.timedelta64(int(embargo_bars * bar_minutes), "m")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n_splits + 1):
        test_start, test_end = edges[i], edges[i + 1]
        if i == n_splits:
            test_mask = (t0 >= test_start) & (t0 <= test_end)
        else:
            test_mask = (t0 >= test_start) & (t0 < test_end)
        train_mask = (t0 < test_start) & (t1 < (test_start - embargo))
        if not test_mask.any() or not train_mask.any():
            continue
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


# --------------------------------------------------------------------------- #
# torch Dataset（薄いラッパ）
# --------------------------------------------------------------------------- #
class SequenceDataset:
    """DataLoader 互換。__getitem__ で (X[seq_len,58], y, w) のテンソルを返す。

    F_norm_by_sym は正規化済み float32 行列（銘柄別）。torch は __getitem__ で遅延 import。
    """

    def __init__(self, F_norm_by_sym: dict[str, np.ndarray], samples: dict, idx: np.ndarray, seq_len: int) -> None:
        self.F = F_norm_by_sym
        self.seq_len = seq_len
        idx = np.asarray(idx, dtype=np.int64)
        self.sym = samples["sym"][idx]
        self.e = samples["e_idx"][idx]
        self.y = samples["y"][idx]
        self.w = samples["w"][idx]

    def __len__(self) -> int:
        return int(self.e.size)

    def __getitem__(self, i: int):
        import torch

        sym = self.sym[i]
        e = int(self.e[i])
        X = self.F[sym][e - self.seq_len + 1: e + 1]
        return (
            torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
            torch.tensor(int(self.y[i]), dtype=torch.long),
            torch.tensor(float(self.w[i]), dtype=torch.float32),
        )


Fold = namedtuple("Fold", ["train", "test", "normalizer", "train_idx", "test_idx"])


def make_folds(
    features_by_sym: dict[str, np.ndarray],
    times_by_sym: dict[str, np.ndarray],
    events_by_sym: dict[str, dict],
    *,
    seq_len: int = SEQ_LEN,
    n_splits: int = N_SPLITS,
    embargo_bars: int = EMBARGO_BARS,
    clip: Optional[float] = CLIP_SIGMA,
    bar_minutes: int = BAR_MINUTES,
) -> list[Fold]:
    """Walk-forward の各 fold について train/test の SequenceDataset と正規化器を作る。

    正規化器は各 fold で train 期間（test_start - embargo より前）のバーのみで fit。
    """
    samples = build_samples(features_by_sym, times_by_sym, events_by_sym, seq_len=seq_len)
    splits = walk_forward_splits(
        samples["t0"], samples["t1"], n_splits=n_splits, embargo_bars=embargo_bars, bar_minutes=bar_minutes
    )
    embargo = np.timedelta64(int(embargo_bars * bar_minutes), "m")

    folds: list[Fold] = []
    for train_idx, test_idx in splits:
        test_start = samples["t0"][test_idx].min()
        cutoff = test_start - embargo
        # train 期間のバーのみで正規化を fit（リーク防止）
        train_rows = []
        for sym, F in features_by_sym.items():
            tt = np.asarray(times_by_sym[sym], dtype="datetime64[ns]")
            rows = np.asarray(F, dtype=np.float64)[tt < cutoff]
            if rows.size:
                rows = rows[np.isfinite(rows).all(axis=1)]
                if rows.size:
                    train_rows.append(rows)
        norm = ZScoreClipNormalizer(clip).fit(np.concatenate(train_rows, axis=0))
        F_norm = {sym: norm.transform(F).astype(np.float32) for sym, F in features_by_sym.items()}
        folds.append(Fold(
            train=SequenceDataset(F_norm, samples, train_idx, seq_len),
            test=SequenceDataset(F_norm, samples, test_idx, seq_len),
            normalizer=norm,
            train_idx=train_idx,
            test_idx=test_idx,
        ))
    return folds


# --------------------------------------------------------------------------- #
# DB ロード
# --------------------------------------------------------------------------- #
async def load_from_db(conn, symbols=("BTC", "ETH")):
    """bar_features と labels を読み、make_folds 入力（features/times/events）を返す。"""
    features_by_sym, times_by_sym, events_by_sym = {}, {}, {}
    cols = ", ".join(FEATURE_NAMES)
    for s in symbols:
        recs = await conn.fetch(
            f"SELECT time, {cols} FROM bar_features WHERE symbol = $1 ORDER BY time", s
        )
        if not recs:
            continue
        times = np.array([r["time"].replace(tzinfo=None) for r in recs], dtype="datetime64[ns]")
        F = np.array([[r[c] if r[c] is not None else np.nan for c in FEATURE_NAMES] for r in recs], dtype=np.float64)
        idx_of = {t: i for i, t in enumerate(times)}

        labs = await conn.fetch(
            "SELECT time, t1, label, sample_weight FROM labels WHERE symbol = $1 ORDER BY time", s
        )
        e_idx, t1, label, weight = [], [], [], []
        for r in labs:
            t = np.datetime64(r["time"].replace(tzinfo=None))
            if t not in idx_of:
                continue
            e_idx.append(idx_of[t])
            t1.append(np.datetime64(r["t1"].replace(tzinfo=None)) if r["t1"] is not None else t)
            label.append(int(r["label"]))
            weight.append(float(r["sample_weight"]) if r["sample_weight"] is not None else 1.0)

        features_by_sym[s] = F
        times_by_sym[s] = times
        events_by_sym[s] = {
            "e_idx": np.array(e_idx, dtype=np.int64),
            "t1": np.array(t1, dtype="datetime64[ns]"),
            "label": np.array(label, dtype=np.int64),
            "weight": np.array(weight, dtype=np.float64),
        }
    return features_by_sym, times_by_sym, events_by_sym
