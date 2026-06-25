"""data/features.py — PC側: Polars で58特徴量をバルク計算.

列順・名称は shared/feature_names.py の FEATURE_NAMES を単一の真実として参照する。
σ は shared/volatility.py、テクニカル指標は shared/technical.py を呼び、train/live で
「同じ入力 → 同じ数値」を構造的に保証する（正規化はしない＝生で保存）。

データ源と特徴量
----------------
* ohlcv_bars            → price/return(11), volatility(6), volume(4), technical(8),
                          temporal(4), microstructure の bar 由来(amihud/roll/kyle/vpin)
* orderbook_snapshots   → OBI(10), spread(3), microstructure の ofi(1)
                          状態量は滞在時間加重平均でバーへ集約。OFI はバー内合計。
* funding_oi            → funding/OI(4)（バー時刻に asof-join）
* Cross-Asset(3)        → 2パス（両銘柄の ret_1 を時刻整合して算出）

窓長は全てバー数（5分足）。時刻特徴は UTC、weekday は ISO(月=1..日=7)、hour=0..23
（live もこの規約で再現すること）。
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import polars as pl

from shared import technical, volatility
from shared.feature_names import FEATURE_NAMES

log = logging.getLogger("aevum.features")

RET_HORIZONS = (1, 5, 15, 30, 60, 240)
_TWO_PI = 2.0 * math.pi
_LN2 = math.log(2.0)

# 板集約後の特徴量（時刻列含まず）
_BOOK_STATE_COLS = [
    "obi_l1", "obi_l5", "obi_l10", "bid_depth_5_log", "ask_depth_5_log",
    "depth_ratio_5", "obi_weighted", "microprice_dev", "bid_slope", "ask_slope",
    "spread_bps",
]
_BOOK_OUT_COLS = _BOOK_STATE_COLS + ["ofi", "spread_z_60", "spread_vol_30"]
_CROSS_COLS = ("cross_corr_60", "cross_beta_60", "cross_ret_spread")


# --------------------------------------------------------------------------- #
# バー（ohlcv）由来の特徴量
# --------------------------------------------------------------------------- #
def compute_bar_features(bars: pl.DataFrame) -> pl.DataFrame:
    """1銘柄の ohlcv_bars から bar 由来の37特徴量を計算（time 列を保持）。"""
    df = bars.sort("time")
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    # numpy 由来（唯一定義）: σ とテクニカル
    sigma = volatility.volatility(close)
    ta = technical.all_technical(high, low, close)

    # 価格・リターン
    df = df.with_columns(
        [(pl.col("close") / pl.col("close").shift(k)).log().alias(f"ret_{k}") for k in RET_HORIZONS]
        + [
            (pl.col("close") / pl.col("open")).log().alias("ret_co"),
            (pl.col("high") / pl.max_horizontal("open", "close")).log().alias("wick_up"),
            (pl.min_horizontal("open", "close") / pl.col("low")).log().alias("wick_dn"),
        ]
    )
    df = df.with_columns([
        ((pl.col("close") - pl.col("low").rolling_min(60))
         / (pl.col("high").rolling_max(60) - pl.col("low").rolling_min(60))).alias("price_pos_60"),
        (pl.col("ret_1") - pl.col("ret_1").shift(1)).alias("ret_accel"),
    ])

    # ボラティリティ
    hl_log_sq = (pl.col("high") / pl.col("low")).log() ** 2
    co_log_sq = (pl.col("close") / pl.col("open")).log() ** 2
    df = df.with_columns([
        ((pl.col("ret_1") ** 2).rolling_sum(30)).sqrt().alias("realized_vol_30"),
        (hl_log_sq.rolling_mean(30) / (4.0 * _LN2)).sqrt().alias("parkinson_30"),
        ((0.5 * hl_log_sq - (2.0 * _LN2 - 1.0) * co_log_sq).rolling_mean(30)).sqrt().alias("garman_klass_30"),
        ((pl.when(pl.col("ret_1") < 0).then(pl.col("ret_1")).otherwise(0.0) ** 2)
         .rolling_mean(30)).sqrt().alias("downside_vol_30"),
    ])
    df = df.with_columns(pl.col("realized_vol_30").rolling_std(60).alias("vol_of_vol_60"))

    # 出来高
    df = df.with_columns([
        pl.col("volume").log1p().alias("volume_log"),
        ((pl.col("volume") - pl.col("volume").rolling_mean(60)) / pl.col("volume").rolling_std(60)).alias("volume_z_60"),
        (pl.col("volume").rolling_mean(5) / pl.col("volume").rolling_mean(30)).alias("volume_ratio_5_30"),
        pl.col("trades").cast(pl.Float64).log1p().alias("trade_count_log"),
    ])

    # microstructure（bar 由来）: amihud / roll_spread / kyle_lambda / vpin_50
    df = df.with_columns([
        (pl.col("ret_1").abs() / (pl.col("close") * pl.col("volume"))).alias("amihud_illiq"),
        (pl.col("ret_1") * pl.col("ret_1").shift(1)).alias("_rlag"),
        (pl.col("ret_1").sign() * pl.col("close") * pl.col("volume")).alias("_sdv"),
        pl.when(pl.col("ret_1") > 0).then(pl.col("volume")).otherwise(0.0).alias("_buy"),
        pl.when(pl.col("ret_1") < 0).then(pl.col("volume")).otherwise(0.0).alias("_sell"),
    ])
    df = df.with_columns([
        (pl.col("_rlag").rolling_mean(30)
         - pl.col("ret_1").rolling_mean(30) * pl.col("ret_1").shift(1).rolling_mean(30)).alias("_cov1"),
        ((pl.col("ret_1") * pl.col("_sdv")).rolling_mean(30)
         - pl.col("ret_1").rolling_mean(30) * pl.col("_sdv").rolling_mean(30)).alias("_covrdv"),
        ((pl.col("_sdv") ** 2).rolling_mean(30) - pl.col("_sdv").rolling_mean(30) ** 2).alias("_vardv"),
    ])
    df = df.with_columns([
        (2.0 * (pl.when(pl.col("_cov1") < 0).then(-pl.col("_cov1")).otherwise(0.0)).sqrt()).alias("roll_spread"),
        (pl.col("_covrdv") / pl.col("_vardv")).alias("kyle_lambda"),
        ((pl.col("_buy").rolling_sum(50) - pl.col("_sell").rolling_sum(50)).abs()
         / pl.col("volume").rolling_sum(50)).alias("vpin_50"),
    ])

    # 時刻（UTC, ISO weekday）
    df = df.with_columns([
        (_TWO_PI * pl.col("time").dt.hour() / 24.0).sin().alias("hour_sin"),
        (_TWO_PI * pl.col("time").dt.hour() / 24.0).cos().alias("hour_cos"),
        (_TWO_PI * pl.col("time").dt.weekday() / 7.0).sin().alias("dow_sin"),
        (_TWO_PI * pl.col("time").dt.weekday() / 7.0).cos().alias("dow_cos"),
    ])

    # numpy 由来を結合
    df = df.with_columns(
        [pl.Series("sigma_ewma", sigma)] + [pl.Series(name, arr) for name, arr in ta.items()]
    )
    return df.drop([c for c in df.columns if c.startswith("_")])


# --------------------------------------------------------------------------- #
# 板（orderbook）由来の特徴量
# --------------------------------------------------------------------------- #
def _book_empty() -> pl.DataFrame:
    schema = {"time": pl.Datetime(time_zone="UTC")}
    schema.update({c: pl.Float64 for c in _BOOK_OUT_COLS})
    return pl.DataFrame(schema=schema)


def compute_book_features(book: pl.DataFrame) -> pl.DataFrame:
    """l2Book スナップショットを5分バーへ集約（状態量=滞在時間加重平均, OFI=バー内合計）。"""
    if book.height == 0:
        return _book_empty()

    df = book.sort("time")

    def bsz(i):
        return pl.col("bid_sz").list.get(i, null_on_oob=True).fill_null(0.0)

    def asz(i):
        return pl.col("ask_sz").list.get(i, null_on_oob=True).fill_null(0.0)

    def bpx(i):
        return pl.col("bid_px").list.get(i, null_on_oob=True)

    def apx(i):
        return pl.col("ask_px").list.get(i, null_on_oob=True)

    bid5 = pl.col("bid_sz").list.head(5).list.sum()
    ask5 = pl.col("ask_sz").list.head(5).list.sum()
    bid10 = pl.col("bid_sz").list.head(10).list.sum()
    ask10 = pl.col("ask_sz").list.head(10).list.sum()
    mid = (bpx(0) + apx(0)) / 2.0
    w_num = sum((bsz(i) - asz(i)) / (i + 1.0) for i in range(5))
    w_den = sum((bsz(i) + asz(i)) / (i + 1.0) for i in range(5))

    # 滞在時間（秒）: 次スナップショットまで、ただしバー終端で打ち切り
    df = df.with_columns([
        pl.col("time").dt.truncate("5m").alias("bar"),
        pl.col("time").shift(-1).alias("_next"),
    ])
    df = df.with_columns((pl.col("bar") + pl.duration(minutes=5)).alias("_bar_end"))
    df = df.with_columns(
        pl.min_horizontal(pl.col("_next").fill_null(pl.col("_bar_end")), pl.col("_bar_end")).alias("_eff_next")
    )
    df = df.with_columns(
        ((pl.col("_eff_next") - pl.col("time")).dt.total_nanoseconds().cast(pl.Float64) / 1e9)
        .clip(lower_bound=0.0).alias("_dwell")
    )

    # スナップショット単位の特徴量
    df = df.with_columns([
        ((bsz(0) - asz(0)) / (bsz(0) + asz(0))).alias("obi_l1"),
        ((bid5 - ask5) / (bid5 + ask5)).alias("obi_l5"),
        ((bid10 - ask10) / (bid10 + ask10)).alias("obi_l10"),
        bid5.log1p().alias("bid_depth_5_log"),
        ask5.log1p().alias("ask_depth_5_log"),
        (bid5 / ask5).alias("depth_ratio_5"),
        (w_num / w_den).alias("obi_weighted"),
        (((bpx(0) * asz(0) + apx(0) * bsz(0)) / (bsz(0) + asz(0)) - mid) / mid).alias("microprice_dev"),
        (bid5 / (bpx(0) - bpx(4))).alias("bid_slope"),
        (ask5 / (apx(4) - apx(0))).alias("ask_slope"),
        ((apx(0) - bpx(0)) / mid * 1e4).alias("spread_bps"),
        bpx(0).alias("_bpx0"), apx(0).alias("_apx0"), bsz(0).alias("_bsz0"), asz(0).alias("_asz0"),
    ])

    # OFI（Cont）: 連続スナップショット間の最良気配変化
    df = df.with_columns([
        pl.col("_bpx0").shift(1).alias("_bpx0p"), pl.col("_apx0").shift(1).alias("_apx0p"),
        pl.col("_bsz0").shift(1).alias("_bsz0p"), pl.col("_asz0").shift(1).alias("_asz0p"),
    ])
    e_bid = (pl.when(pl.col("_bpx0") >= pl.col("_bpx0p")).then(pl.col("_bsz0")).otherwise(0.0)
             - pl.when(pl.col("_bpx0") <= pl.col("_bpx0p")).then(pl.col("_bsz0p")).otherwise(0.0))
    e_ask = (pl.when(pl.col("_apx0") <= pl.col("_apx0p")).then(pl.col("_asz0")).otherwise(0.0)
             - pl.when(pl.col("_apx0") >= pl.col("_apx0p")).then(pl.col("_asz0p")).otherwise(0.0))
    df = df.with_columns((e_bid - e_ask).alias("_ofi_inc"))

    # バー集約
    agg = (
        df.group_by("bar")
        .agg(
            [((pl.col(c) * pl.col("_dwell")).sum() / pl.col("_dwell").sum()).alias(c) for c in _BOOK_STATE_COLS]
            + [pl.col("_ofi_inc").sum().alias("ofi")]
        )
        .sort("bar")
        .rename({"bar": "time"})
    )
    agg = agg.with_columns([
        ((pl.col("spread_bps") - pl.col("spread_bps").rolling_mean(60)) / pl.col("spread_bps").rolling_std(60)).alias("spread_z_60"),
        pl.col("spread_bps").rolling_std(30).alias("spread_vol_30"),
    ])
    return agg.select(["time", *_BOOK_OUT_COLS])


# --------------------------------------------------------------------------- #
# Funding / OI
# --------------------------------------------------------------------------- #
def compute_funding_features(bar_times: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    """バー時刻に最新 funding_oi を asof-join して funding/OI(4) を算出。"""
    cols = ["time", "funding_rate", "funding_z_60", "oi_log", "oi_change"]
    if funding.height == 0:
        out = bar_times.select("time").with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in cols[1:]]
        )
        return out.select(cols)

    f = funding.select("time", "funding_rate", "open_interest").sort("time")
    out = bar_times.select("time").sort("time").join_asof(f, on="time", strategy="backward")
    out = out.with_columns([
        ((pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(60)) / pl.col("funding_rate").rolling_std(60)).alias("funding_z_60"),
        pl.col("open_interest").log1p().alias("oi_log"),
        (pl.col("open_interest") / pl.col("open_interest").shift(1)).log().alias("oi_change"),
    ])
    return out.select(cols)


# --------------------------------------------------------------------------- #
# Cross-Asset（2パス目）
# --------------------------------------------------------------------------- #
def compute_cross_features(self_ret: pl.DataFrame, other_ret: pl.DataFrame) -> pl.DataFrame:
    """self/other の ret_1 を時刻整合し、60バーの相関/ベータ/スプレッドを算出。"""
    w = 60
    m = (
        self_ret.select("time", pl.col("ret_1").alias("_rs"))
        .join(other_ret.select("time", pl.col("ret_1").alias("_ro")), on="time", how="inner")
        .sort("time")
    )
    m = m.with_columns([
        pl.col("_rs").rolling_mean(w).alias("_ms"),
        pl.col("_ro").rolling_mean(w).alias("_mo"),
        (pl.col("_rs") * pl.col("_ro")).rolling_mean(w).alias("_mso"),
        (pl.col("_rs") ** 2).rolling_mean(w).alias("_mss"),
        (pl.col("_ro") ** 2).rolling_mean(w).alias("_moo"),
    ])
    m = m.with_columns([
        (pl.col("_mso") - pl.col("_ms") * pl.col("_mo")).alias("_cov"),
        (pl.col("_mss") - pl.col("_ms") ** 2).alias("_vs"),
        (pl.col("_moo") - pl.col("_mo") ** 2).alias("_vo"),
    ])
    m = m.with_columns([
        (pl.col("_cov") / (pl.col("_vs") * pl.col("_vo")).sqrt()).alias("cross_corr_60"),
        (pl.col("_cov") / pl.col("_vo")).alias("cross_beta_60"),
        (pl.col("_rs") - pl.col("_ro")).alias("cross_ret_spread"),
    ])
    return m.select(["time", *_CROSS_COLS])


# --------------------------------------------------------------------------- #
# 統合（パス1 → パス2）
# --------------------------------------------------------------------------- #
def _assemble_symbol(symbol: str, bars: pl.DataFrame, book: pl.DataFrame, funding: pl.DataFrame) -> pl.DataFrame:
    barf = compute_bar_features(bars)
    bookf = compute_book_features(book)
    fund = compute_funding_features(barf.select("time"), funding)
    out = barf.join(bookf, on="time", how="left").join(fund, on="time", how="left")
    return out.with_columns(pl.lit(symbol).alias("symbol"))


def compute_features(
    bars_by_symbol: dict[str, pl.DataFrame],
    book_by_symbol: Optional[dict[str, pl.DataFrame]] = None,
    funding_by_symbol: Optional[dict[str, pl.DataFrame]] = None,
) -> dict[str, pl.DataFrame]:
    """全銘柄の58特徴量を2パスで計算。各 DataFrame は [symbol, time, *FEATURE_NAMES]。"""
    book_by_symbol = book_by_symbol or {}
    funding_by_symbol = funding_by_symbol or {}

    pass1 = {
        s: _assemble_symbol(s, bars, book_by_symbol.get(s, _book_empty()),
                            funding_by_symbol.get(s, pl.DataFrame(schema={"time": pl.Datetime(time_zone="UTC")})))
        for s, bars in bars_by_symbol.items()
    }

    result: dict[str, pl.DataFrame] = {}
    syms = list(pass1)
    for s in syms:
        df = pass1[s]
        others = [o for o in syms if o != s]
        if others:
            cross = compute_cross_features(df.select("time", "ret_1"), pass1[others[0]].select("time", "ret_1"))
            df = df.join(cross, on="time", how="left")
        else:
            df = df.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in _CROSS_COLS])
        result[s] = df.select(["symbol", "time", *FEATURE_NAMES])
    return result


# --------------------------------------------------------------------------- #
# DB I/O
# --------------------------------------------------------------------------- #
def _build_insert_sql() -> str:
    cols = ["symbol", "time", *FEATURE_NAMES]
    ph = ", ".join(f"${i + 1}" for i in range(len(cols)))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in FEATURE_NAMES)
    return (
        f"INSERT INTO bar_features ({', '.join(cols)}) VALUES ({ph}) "
        f"ON CONFLICT (symbol, time) DO UPDATE SET {updates}"
    )


BAR_FEATURES_INSERT_SQL = _build_insert_sql()


async def fetch_bars(conn, symbol: str) -> pl.DataFrame:
    recs = await conn.fetch(
        "SELECT time, open, high, low, close, volume, trades FROM ohlcv_bars WHERE symbol = $1 ORDER BY time",
        symbol,
    )
    return pl.DataFrame([dict(r) for r in recs]) if recs else pl.DataFrame(
        schema={"time": pl.Datetime(time_zone="UTC"), "open": pl.Float64, "high": pl.Float64,
                "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64, "trades": pl.Int64}
    )


async def fetch_book(conn, symbol: str) -> pl.DataFrame:
    recs = await conn.fetch(
        "SELECT time, bid_px, bid_sz, ask_px, ask_sz FROM orderbook_snapshots WHERE symbol = $1 ORDER BY time",
        symbol,
    )
    return pl.DataFrame([dict(r) for r in recs]) if recs else pl.DataFrame()


async def fetch_funding(conn, symbol: str) -> pl.DataFrame:
    recs = await conn.fetch(
        "SELECT time, funding_rate, open_interest FROM funding_oi WHERE symbol = $1 ORDER BY time",
        symbol,
    )
    return pl.DataFrame([dict(r) for r in recs]) if recs else pl.DataFrame()


async def store_features(conn, df: pl.DataFrame) -> None:
    rows = df.rows()
    if rows:
        await conn.executemany(BAR_FEATURES_INSERT_SQL, rows)


async def run(symbols=("BTC", "ETH")) -> None:
    import asyncpg

    from data.ingestion import resolve_dsn

    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            bars = {s: await fetch_bars(conn, s) for s in symbols}
            book = {s: await fetch_book(conn, s) for s in symbols}
            funding = {s: await fetch_funding(conn, s) for s in symbols}
            feats = compute_features(bars, book, funding)
            for s, df in feats.items():
                await store_features(conn, df)
                log.info("features stored: symbol=%s rows=%d", s, df.height)
    finally:
        await pool.close()


def main() -> None:
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()
