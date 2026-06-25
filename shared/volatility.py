"""shared/volatility.py — σ（ボラティリティ）計算式の【唯一の定義】.

Aevum 全体で σ を計算してよいのはこのモジュールだけ。CLAUDE.md の最重要不変条件:

    labels.py      … Triple-Barrier のバリアサイズ決定
    execution.py   … ライブのバリア幅
    監視UI         … σ 表示

の3箇所は、必ず `volatility()` / `latest()` を呼ぶこと。別々に式を再実装しては
ならない（式がズレると致命的な失敗モードになる）。スケール（保有期間 √t 変換）も
`scale_to_horizon()` に集約し、二重実装を禁止する。

確定仕様
--------
* 推定法   : EWMA 標準偏差（不偏 / adjust=True）。AFML `getDailyVol` 準拠であり、
             pandas `Series.ewm(span, adjust=True).std()`（bias=False）と一致する。
* リターン : 対数リターン  r_t = ln(p_t / p_{t-1})
* スケール : 1バーあたり σ を返す。保有期間/倍率へのスケールは呼び出し側が行う。

実装方針
--------
* 依存は numpy のみ。PC(Polars) / Pi(numpy/pandas) / UI のどこからでも同一結果。
* バッチ計算が真の定義。ライブは十分長いバッファに対して同じ関数を呼ぶことで一致
  させる（`recommended_min_history()` 参照）。

参考: López de Prado, *Advances in Financial Machine Learning*, Snippet 3.1
      (getDailyVol — daily volatility estimated via EWM std of returns).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "DEFAULT_SPAN",
    "DEFAULT_MIN_PERIODS",
    "log_returns",
    "ewma_std",
    "volatility",
    "latest",
    "recommended_min_history",
    "scale_to_horizon",
]

ArrayLike = np.ndarray | Sequence[float]

# AFML getDailyVol の既定スパン。全呼び出し側はこの既定値を共有する。
DEFAULT_SPAN: int = 100
# σ を確定させるのに必要な最小リターン本数。
# 不偏 EWMA 分散はリターン2本以上で初めて定義できる（1本では脱バイアス係数が 0 除算）。
DEFAULT_MIN_PERIODS: int = 2


def _as_1d_float(x: ArrayLike, *, name: str) -> np.ndarray:
    """入力を 1 次元 float64 配列へ変換（コピー）。2 次元以上は拒否。"""
    arr = np.array(x, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got ndim={arr.ndim}")
    return arr


def log_returns(close: ArrayLike) -> np.ndarray:
    """対数リターン r_t = ln(p_t / p_{t-1}) を返す。

    返り値は入力と同じ長さで、`r[0]` は NaN（直前バーが無いため）。
    価格は正の有限値であること（<=0 や NaN/Inf は ValueError）。
    """
    p = _as_1d_float(close, name="close")
    out = np.full(p.shape, np.nan, dtype=np.float64)
    if p.size == 0:
        return out
    if not np.all(np.isfinite(p)):
        raise ValueError("close contains NaN/Inf; clean the series first")
    if np.any(p <= 0.0):
        raise ValueError("close must be strictly positive for log returns")
    out[1:] = np.log(p[1:] / p[:-1])
    return out


def ewma_std(
    x: ArrayLike,
    span: int,
    *,
    min_periods: int = DEFAULT_MIN_PERIODS,
    bias: bool = False,
) -> np.ndarray:
    """指数加重移動標準偏差（adjust=True）。

    pandas `Series(x).ewm(span=span, adjust=True, min_periods=min_periods).std(bias=bias)`
    と一致する。既定の `bias=False` は不偏推定（脱バイアス係数 s1^2/(s1^2-s2)）で、
    AFML の `ewm(span).std()` と同じ。

    入力 `x` は有限値の 1 次元配列（NaN/Inf 不可）。出力は `x` と同じ長さで、
    有効サンプル数が `min_periods` 未満の要素は NaN。

    重み（adjust=True）: 位置 t において w_j = decay**j （j=0 が最新、decay=1-alpha,
    alpha=2/(span+1)）。累積和を O(n) の再帰で計算する。
    """
    arr = _as_1d_float(x, name="x")
    n = arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    if not np.all(np.isfinite(arr)):
        raise ValueError("ewma_std input contains NaN/Inf")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")

    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha

    wsum_x = 0.0   # Σ w_j * x
    wsum_xx = 0.0  # Σ w_j * x^2
    s1 = 0.0       # Σ w_j
    s2 = 0.0       # Σ w_j^2
    for t in range(n):
        xt = arr[t]
        wsum_x = xt + decay * wsum_x
        wsum_xx = xt * xt + decay * wsum_xx
        s1 = 1.0 + decay * s1
        s2 = 1.0 + (decay * decay) * s2

        count = t + 1
        if count < min_periods:
            continue

        mean = wsum_x / s1
        biased_var = wsum_xx / s1 - mean * mean
        if biased_var < 0.0:  # 浮動小数点誤差で僅かに負になり得る
            biased_var = 0.0

        if bias:
            var = biased_var
        else:
            denom = s1 * s1 - s2
            if denom <= 0.0:
                # 有効サンプルが実質1本（脱バイアス不能）→ NaN のまま残す
                continue
            var = biased_var * (s1 * s1) / denom

        out[t] = math.sqrt(var)
    return out


def volatility(
    close: ArrayLike,
    span: int = DEFAULT_SPAN,
    *,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> np.ndarray:
    """【唯一の定義】1バーあたり σ = 対数リターンの EWMA 標準偏差。

    返り値は `close` と同じ長さ（インデックス整合）。`σ[0]` は NaN（リターン無し）。
    先頭の有効リターン本数が `min_periods` 未満の位置も NaN。

    保有期間や倍率（pt_sl）へのスケールは呼び出し側で行うこと（per-bar σ）。
    保有期間スケールが必要なら `scale_to_horizon()` を使う。
    """
    p = _as_1d_float(close, name="close")
    sigma = np.full(p.shape, np.nan, dtype=np.float64)
    if p.size < 2:
        return sigma
    r = log_returns(p)                # r[0]=nan, r[1:] は有限（価格は正で検証済み）
    s = ewma_std(r[1:], span, min_periods=min_periods)  # 長さ n-1
    sigma[1:] = s
    return sigma


def latest(
    close: ArrayLike,
    span: int = DEFAULT_SPAN,
    *,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> float:
    """直近バーの σ（スカラ）。execution.py / 監視UI 用。

    `close` は直近クローズのローリングバッファ。EWMA を学習時（バッチ）と一致させる
    には `recommended_min_history(span)` 以上の長さを渡すこと。値が未確定なら NaN。
    """
    s = volatility(close, span, min_periods=min_periods)
    return float(s[-1]) if s.size else float("nan")


def recommended_min_history(span: int, tol: float = 1e-6) -> int:
    """ライブの σ をバッチ計算と `tol` 以内で一致させるのに必要な最小バー数。

    adjust=True の EWMA は全履歴の幾何重みで正規化されるため、有限バッファだと
    最古重み decay**k の打ち切り誤差が出る。decay**k <= tol となる k を求め、
    リターン1本ぶん +1 した本数を返す（最低 DEFAULT_MIN_PERIODS+1）。
    """
    span = int(span)
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    if not (0.0 < tol < 1.0):
        raise ValueError(f"tol must be in (0, 1), got {tol}")
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    if decay <= 0.0:
        k = 1
    else:
        k = int(math.ceil(math.log(tol) / math.log(decay)))
    return max(k + 1, DEFAULT_MIN_PERIODS + 1)


def scale_to_horizon(sigma, horizon: int):
    """per-bar σ を保有期間（horizon バー）へ √t スケール: σ_h = σ * sqrt(horizon)。

    labels.py / execution.py がバリア幅を保有期間へ合わせる際は必ずこれを使う
    （スケール式もここで統一し、二重実装を禁止する）。`sigma` はスカラ/配列可。
    スカラ入力なら float、配列入力なら ndarray を返す。
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    factor = math.sqrt(float(horizon))
    if np.isscalar(sigma):
        return float(sigma) * factor
    return np.asarray(sigma, dtype=np.float64) * factor
