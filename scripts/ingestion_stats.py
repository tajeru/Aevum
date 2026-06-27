"""scripts/ingestion_stats.py — 生データ取り込みの量・頻度を実測（段階2 goal #3）.

2 時点のサンプリング差分から、テーブル別に rows/sec・rows/day と、Timescale の
hypertable 実サイズ増分（≒ bytes/day）を出す。orderbook_snapshots（全更新保存）の
日次見積もり検証用。ingestion 稼働中に別プロセスで実行する。

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    .\.venv\Scripts\python.exe scripts/ingestion_stats.py --window 120
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.ingestion import TABLES, resolve_dsn  # noqa: E402


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}PB"


async def _hypertable_size(conn, table: str) -> int:
    """Timescale hypertable の実サイズ（チャンク込み）。非 hypertable は通常サイズ。"""
    try:
        return int(await conn.fetchval("SELECT hypertable_size($1::regclass)", table))
    except Exception:
        return int(await conn.fetchval("SELECT pg_total_relation_size($1::regclass)", table))


async def _sample(conn) -> dict:
    out = {}
    for t in TABLES:
        cnt = await conn.fetchval(f"SELECT count(*) FROM {t}")
        size = await _hypertable_size(conn, t)
        tmin = await conn.fetchval(f"SELECT min(time) FROM {t}")
        tmax = await conn.fetchval(f"SELECT max(time) FROM {t}")
        out[t] = {"rows": cnt, "size": size, "tmin": tmin, "tmax": tmax}
    return out


async def main(window: float) -> None:
    dsn = resolve_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        a = await _sample(conn)
        print(f"sampling for {window:.0f}s ...")
        await asyncio.sleep(window)
        b = await _sample(conn)
    finally:
        await conn.close()

    print(f"\n=== ingestion stats over {window:.0f}s ===")
    for t in TABLES:
        d = b[t]["rows"] - a[t]["rows"]
        rate = d / window if window else 0.0
        per_day = rate * 86400
        dsize = b[t]["size"] - a[t]["size"]
        bytes_per_row = (dsize / d) if d else 0.0
        bytes_per_day = (dsize / window * 86400) if window else 0.0
        print(
            f"{t:22s} +{d:>6d} rows  {rate:6.2f}/s  ~{per_day:>12,.0f}/day | "
            f"total={b[t]['rows']:>8d} size={_human(b[t]['size']):>9s} "
            f"Δsize={_human(dsize):>9s} (~{_human(bytes_per_day)}/day, {bytes_per_row:,.0f} B/row) | "
            f"latest={b[t]['tmax']}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=120.0, help="サンプリング秒数")
    args = ap.parse_args()
    asyncio.run(main(args.window))
