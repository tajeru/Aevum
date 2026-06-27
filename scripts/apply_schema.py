"""scripts/apply_schema.py — schema_v1.sql を DB へ適用（psql 不要・asyncpg 経由）.

ローカル psql が無い環境用。接続 DSN は ingestion.resolve_dsn()（AEVUM_DB_DSN 優先）。
schema_v1.sql は IF NOT EXISTS / if_not_exists で冪等なので再実行可。

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    .\.venv\Scripts\python.exe scripts/apply_schema.py
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

import asyncpg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.ingestion import resolve_dsn  # noqa: E402

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[1] / "schema" / "schema_v1.sql"


async def main() -> None:
    dsn = resolve_dsn()
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    print(f"applying {SCHEMA_PATH} -> {dsn}")
    conn = await asyncpg.connect(dsn)
    try:
        # 引数なし execute は simple-query プロトコルで複数文を一括実行できる。
        await conn.execute(sql)
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        ext = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
        )
    finally:
        await conn.close()
    print(f"timescaledb extension: {ext}")
    print(f"tables ({len(tables)}): {[t['tablename'] for t in tables]}")


if __name__ == "__main__":
    asyncio.run(main())
