"""scripts/seed_dev.py — 段階1の e2e 用に少量のテストデータを投入（dev 専用）.

build_snapshot が読む全要素を網羅して投入する:
  * ohlcv_bars        … PriceChart のローソク（BTC 120本）
  * model_predictions … SignalPanel/σ の最新予測（BTC）
  * bar_features      … live features（BTC 最新行・代表列のみ）
  * orders            … ブラケット（entry=limit, take_profit=limit, stop_loss=market+triggerPx）
                        → chart.barriers / positions.slLevel / barsHeld(entry注文時刻) の源
  * positions         … 建玉（BTC ロング / ETH ショート）
  * labels            … 学習ラベル少量（snapshot 非依存だが疎通確認用）

冪等性: 各テーブルを DELETE してから投入（dev 専用の全リセット）。
接続: ingestion.resolve_dsn()（AEVUM_DB_DSN 優先）。

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    .\.venv\Scripts\python.exe scripts/seed_dev.py
"""
from __future__ import annotations

import asyncio
import math
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.ingestion import resolve_dsn  # noqa: E402

BAR = timedelta(minutes=5)
N_BARS = 120


def _d(x: float) -> Decimal:
    """NUMERIC 列用（asyncpg は numeric に Decimal を要求）。"""
    return Decimal(str(x))


def _floor_5min(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)


def _gen_candles(symbol: str, n: int, end_time: datetime, start_price: float):
    """決定論的ローソク（seed 固定 LCG）。FLOAT8 列なので float のまま。"""
    seed = 0x9E3779B1

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return seed / 0xFFFFFFFF

    rows = []
    close = start_price
    start_time = end_time - (n - 1) * BAR
    for i in range(n):
        drift = math.sin(i / 9) * 0.0008
        shock = (rnd() - 0.5) * 0.006
        o = close
        close = o * (1 + drift + shock)
        hi = max(o, close) * (1 + rnd() * 0.0025)
        lo = min(o, close) * (1 - rnd() * 0.0025)
        vol = 10.0 + rnd() * 40.0
        rows.append((
            symbol, start_time + i * BAR,
            round(o, 1), round(hi, 1), round(lo, 1), round(close, 1),
            round(vol, 3), int(50 + rnd() * 100),
        ))
    return rows


async def main() -> None:
    dsn = resolve_dsn()
    now = _floor_5min(datetime.now(timezone.utc))
    print(f"seeding -> {dsn} (latest bar = {now.isoformat()})")

    candles = _gen_candles("BTC", N_BARS, now, 64000.0)
    last_close = candles[-1][5]  # close of latest bar

    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            # --- dev 全リセット --- #
            for t in ("orders", "positions", "model_predictions",
                      "bar_features", "labels", "ohlcv_bars"):
                await conn.execute(f"DELETE FROM {t}")

            # --- ohlcv_bars (BTC) --- #
            await conn.executemany(
                "INSERT INTO ohlcv_bars (symbol, time, open, high, low, close, volume, trades) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                candles,
            )

            # --- model_predictions (BTC 最新) --- #
            await conn.execute(
                "INSERT INTO model_predictions "
                "(symbol, time, model_version, prob_down, prob_flat, prob_up, signal, sigma) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                "BTC", now, "seed-v1", 0.18, 0.29, 0.53, 1, 0.0041,
            )

            # --- bar_features (BTC 最新・代表列のみ。他列は NULL 可) --- #
            await conn.execute(
                "INSERT INTO bar_features "
                "(symbol, time, sigma_ewma, obi_l5, rsi_14, macd_hist, spread_bps, funding_rate) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                "BTC", now, 0.0041, 0.137, 58.3, 14.2, 1.8, 0.00012,
            )

            # --- orders: ブラケット --- #
            # 非対称設計: entry/take_profit=limit, stop_loss=market(+triggerPx in price)。
            btc_entry_t = now - timedelta(minutes=30)   # barsHeld = 6
            eth_entry_t = now - timedelta(minutes=45)    # barsHeld = 9
            order_rows = [
                # BTC ロングのブラケット
                ("BTC", btc_entry_t, "buy", "entry", "limit", _d(64210.0), _d(0.15), "filled", True),
                ("BTC", btc_entry_t, "sell", "take_profit", "limit", _d(65800.0), _d(0.15), "open", True),
                ("BTC", btc_entry_t, "sell", "stop_loss", "market", _d(63200.0), _d(0.15), "open", True),
                # ETH ショートのブラケット（SL は建値の上）
                ("ETH", eth_entry_t, "sell", "entry", "limit", _d(3420.0), _d(1.2), "filled", True),
                ("ETH", eth_entry_t, "buy", "stop_loss", "market", _d(3510.0), _d(1.2), "open", True),
            ]
            await conn.executemany(
                "INSERT INTO orders "
                "(symbol, time, side, intent, order_type, price, size, status, risk_passed) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)",
                order_rows,
            )

            # --- positions --- #
            btc_upnl = round((last_close - 64210.0) * 0.15, 2)
            position_rows = [
                ("BTC", now, _d(0.15), _d(64210.0), _d(last_close), _d(btc_upnl),
                 _d(0.0), 3.0, _d(48230.0), _d(3210.5)),
                ("ETH", now, _d(-1.2), _d(3420.0), _d(3398.0), _d(26.4),
                 _d(0.0), 2.0, _d(4120.0), _d(2038.8)),
            ]
            await conn.executemany(
                "INSERT INTO positions "
                "(symbol, time, size, entry_price, mark_price, unrealized_pnl, realized_pnl, "
                " leverage, liquidation_px, margin_used) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                position_rows,
            )

            # --- labels（疎通確認用に少量） --- #
            t0 = now - 60 * BAR
            label_rows = [
                ("BTC", t0, 1, 0.012, 0.0040, 65000.0, 63500.0, 1.0, 1.0,
                 48, t0 + 48 * BAR, t0 + 20 * BAR, "pt", 0.8),
                ("BTC", t0 + BAR, -1, -0.009, 0.0042, 65100.0, 63600.0, 1.0, 1.0,
                 48, t0 + 49 * BAR, t0 + 31 * BAR, "sl", 0.7),
            ]
            await conn.executemany(
                "INSERT INTO labels "
                "(symbol, time, label, ret, sigma, pt_level, sl_level, pt_mult, sl_mult, "
                " horizon_bars, t1, touch_time, touch_barrier, sample_weight) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",
                label_rows,
            )

        # --- 件数サマリ --- #
        for t in ("ohlcv_bars", "model_predictions", "bar_features",
                  "orders", "positions", "labels"):
            c = await conn.fetchval(f"SELECT count(*) FROM {t}")
            print(f"  {t}: {c}")
    finally:
        await conn.close()
    print("seed done.")


if __name__ == "__main__":
    asyncio.run(main())
