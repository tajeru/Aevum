"""shared/technical.py — テクニカル指標の【唯一の定義】(numpy).

σ を shared/volatility.py に一元化したのと同じ思想で、テクニカル指標の計算式も
ここを唯一の真実とする。PC(features.py / Polars パイプライン)も Pi(live / 逐次)も
このモジュールを呼び、train/live で「同じ入力 → 同じ数値」を構造的に保証する。

確定仕様
--------
* RSI / ATR / ADX : Wilder 平滑（RMA, alpha=1/period, SMA シード）
* MACD            : EMA(fast=12, slow=26, signal=9)（adjust=False の再帰 EMA）
* Stochastic %K   : 直近 period の高安レンジ内での終値位置
* Bollinger %B    : (close - lower) / (upper - lower)（period=20, k=2, 母標準偏差）

すべて入力と同じ長さの配列を返し、ウォームアップ期間は NaN。
features.py の technical カテゴリ（8列）は all_technical() が一括で返す。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ema",
    "wilder_rma",
    "rsi",
    "macd",
    "true_range",
    "atr",
    "stoch_k",
    "adx",
    "bb_pctb",
    "all_technical",
]


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def ema(x, span: int) -> np.ndarray:
    """再帰 EMA（adjust=False, 先頭値シード）。pandas ewm(span, adjust=False).mean() と一致。"""
    x = _arr(x)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_rma(x, period: int) -> np.ndarray:
    """Wilder の平滑移動平均（RMA）。最初の period 個の単純平均をシードに再帰。

    先頭の NaN（ウォームアップ）は飛ばし、最初の有限値以降が連続して有限である前提
    （本モジュール内の入力は条件を満たす）。
    """
    x = _arr(x)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 1 or n < period:
        return out
    finite = np.where(np.isfinite(x))[0]
    if finite.size == 0:
        return out
    f = int(finite[0])
    if n - f < period:
        return out
    seed_idx = f + period - 1
    out[seed_idx] = float(np.mean(x[f:f + period]))
    for i in range(seed_idx + 1, n):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def rsi(close, period: int = 14) -> np.ndarray:
    """Wilder RSI。単調増加→100、単調減少→0、フラット→50。"""
    close = _arr(close)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out
    delta = np.diff(close)                      # 長さ n-1（delta[d] は価格 index d+1）
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = wilder_rma(gain, period)
    al = wilder_rma(loss, period)
    for d in range(period - 1, n - 1):
        a, l = ag[d], al[d]
        if not (np.isfinite(a) and np.isfinite(l)):
            continue
        if l == 0.0:
            out[d + 1] = 100.0 if a > 0.0 else 50.0
        else:
            rs = a / l
            out[d + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD。返り値 (macd_line, signal, hist)。EMA(fast)-EMA(slow), signal=EMA(line)。"""
    close = _arr(close)
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def true_range(high, low, close) -> np.ndarray:
    """True Range: max(H-L, |H-prevC|, |L-prevC|)。先頭は H-L。"""
    high, low, close = _arr(high), _arr(low), _arr(close)
    n = close.size
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    return tr


def atr(high, low, close, period: int = 14, *, normalize: bool = True) -> np.ndarray:
    """ATR（Wilder 平滑の True Range）。normalize=True で close で割る（atr_14 特徴）。"""
    a = wilder_rma(true_range(high, low, close), period)
    if normalize:
        a = a / _arr(close)
    return a


def stoch_k(high, low, close, period: int = 14) -> np.ndarray:
    """Stochastic %K = (close - LL) / (HH - LL) * 100。レンジ0は50。"""
    high, low, close = _arr(high), _arr(low), _arr(close)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = high[i - period + 1:i + 1].max()
        ll = low[i - period + 1:i + 1].min()
        rng = hh - ll
        out[i] = (close[i] - ll) / rng * 100.0 if rng > 0 else 50.0
    return out


def adx(high, low, close, period: int = 14) -> np.ndarray:
    """Wilder ADX。+DM/-DM と TR を Wilder 平滑し DX→ADX。最初の有効値は ~2*period-1。"""
    high, low, close = _arr(high), _arr(low), _arr(close)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * period:
        return out
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus_dm = np.where((up > dn) & (up > 0.0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    tr = true_range(high, low, close)[1:]

    sm_plus = wilder_rma(plus_dm, period)
    sm_minus = wilder_rma(minus_dm, period)
    sm_tr = wilder_rma(tr, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        di_plus = 100.0 * sm_plus / sm_tr
        di_minus = 100.0 * sm_minus / sm_tr
        denom = di_plus + di_minus
        dx = 100.0 * np.abs(di_plus - di_minus) / denom
    dx = np.where(denom == 0.0, 0.0, dx)        # denom==0 → DX=0（nan は維持）
    adx_vals = wilder_rma(dx, period)
    out[1:] = adx_vals
    return out


def bb_pctb(close, period: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger %B = (close - lower) / (upper - lower)。母標準偏差(ddof=0)。"""
    close = _arr(close)
    n = close.size
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        w = close[i - period + 1:i + 1]
        m = w.mean()
        s = w.std(ddof=0)
        upper, lower = m + k * s, m - k * s
        rng = upper - lower
        out[i] = (close[i] - lower) / rng if rng > 0 else 0.5
    return out


def all_technical(high, low, close) -> dict[str, np.ndarray]:
    """technical カテゴリの8特徴量を一括計算（キーは shared.feature_names と一致）。"""
    line, sig, hist = macd(close)
    return {
        "rsi_14": rsi(close, 14),
        "macd_line": line,
        "macd_signal": sig,
        "macd_hist": hist,
        "bb_pctb_20": bb_pctb(close, 20),
        "atr_14": atr(high, low, close, 14, normalize=True),
        "stoch_k_14": stoch_k(high, low, close, 14),
        "adx_14": adx(high, low, close, 14),
    }
