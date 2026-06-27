"""check_db.py — DB の各テーブルの件数・時刻レンジを確認する（dev 調査用）.

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    ./.venv/Scripts/python.exe check_db.py
"""
import asyncio
import os

import asyncpg

TABLES = (
    "ohlcv_bars",
    "orderbook_snapshots",
    "funding_oi",
    "bar_features",
    "labels",
    "model_predictions",
)


async def main():
    dsn = os.environ["AEVUM_DB_DSN"]
    conn = await asyncpg.connect(dsn)
    try:
        for t in TABLES:
            n = await conn.fetchval(f"select count(*) from {t}")
            mn = await conn.fetchval(f"select min(time) from {t}")
            mx = await conn.fetchval(f"select max(time) from {t}")
            print(f"{t:22s} {n:>8d}  {mn}  ..  {mx}")

        print("--- per symbol ohlcv_bars ---")
        for s in ("BTC", "ETH"):
            n = await conn.fetchval(
                "select count(*) from ohlcv_bars where symbol=$1", s
            )
            print(f"  {s}: {n}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
