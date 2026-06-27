"""scripts/run_features.py — features.py を実DBデータに対して走らせ bar_features を生成.

features 接続ランナー（features.py 本体は無改修）。data.features の
fetch_* / compute_features / store_features をそのまま呼び、以下を実測・報告する:

* バルク計算の壁時計時間（Polars が本番規模で動くか）
* 各銘柄の入力規模（bars / book / funding 件数）
* 出力 bar_features の行数と、カテゴリ別の有効値（非NULL）率
  - 板系（OBI/Spread/Microstructure-ofi）は板蓄積待ちで NULL/疎が正常
* ウォームアップ NaN の本数（履歴依存特徴）

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    ./.venv/Scripts/python.exe scripts/run_features.py
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import asyncpg
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.features import (  # noqa: E402
    compute_features,
    fetch_bars,
    fetch_book,
    fetch_funding,
    store_features,
)
from data.ingestion import resolve_dsn  # noqa: E402
from shared.feature_names import FEATURE_CATEGORIES, FEATURE_NAMES  # noqa: E402


def _nonnull_rate(df: pl.DataFrame, cols: list[str]) -> tuple[int, int]:
    """指定列について (全セル数, 非NULL かつ非NaN セル数) を返す。"""
    total = 0
    good = 0
    h = df.height
    for c in cols:
        s = df[c]
        total += h
        # NULL と NaN の両方を欠損として数える
        good += h - int(s.is_null().sum()) - int(s.is_nan().sum())
    return total, good


async def main(symbols: tuple[str, ...], store: bool) -> None:
    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=4)
    try:
        async with pool.acquire() as conn:
            bars, book, funding = {}, {}, {}
            for s in symbols:
                bars[s] = await fetch_bars(conn, s)
                book[s] = await fetch_book(conn, s)
                funding[s] = await fetch_funding(conn, s)
                print(
                    f"input {s:4s}: bars={bars[s].height:>6d}  "
                    f"book={book[s].height:>6d}  funding={funding[s].height:>6d}"
                )

            t0 = time.perf_counter()
            feats = compute_features(bars, book, funding)
            dt = time.perf_counter() - t0
            total_bars = sum(b.height for b in bars.values())
            print(
                f"\ncompute_features: {dt * 1000:.1f} ms for {total_bars} bars "
                f"({len(symbols)} symbols, 2-pass cross)  "
                f"= {dt / max(total_bars, 1) * 1e6:.1f} us/bar"
            )

            for s, df in feats.items():
                print(f"\n=== {s}: rows={df.height} cols={df.width} ===")
                for cat, cols in FEATURE_CATEGORIES.items():
                    total, good = _nonnull_rate(df, list(cols))
                    pct = 100.0 * good / total if total else 0.0
                    print(f"  {cat:14s} {good:>6d}/{total:<6d} non-null ({pct:5.1f}%)")
                if store:
                    await store_features(conn, df)
                    print(f"  -> stored {df.height} rows into bar_features")

            if store:
                n = await conn.fetchval("select count(*) from bar_features")
                print(f"\nbar_features total rows now: {n}")
    finally:
        await pool.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC,ETH", help="カンマ区切り銘柄")
    ap.add_argument("--no-store", action="store_true", help="計算のみ（DB書き込みなし）")
    args = ap.parse_args()
    syms = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    asyncio.run(main(syms, store=not args.no_store))
