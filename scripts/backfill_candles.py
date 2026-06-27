"""scripts/backfill_candles.py — REST candleSnapshot で過去ローソクを ohlcv_bars へ投入.

WS 購読は履歴をバックフィルしないため、過去 OHLCV を REST で取り込む（過去データ投入。
features 本体には触れない）。板 / funding は過去取得不可なので OHLCV のみが対象
（OHLCV系特徴は過去も計算可、板系特徴=OBI/Spread/Microstructure はライブ以降のみ）。

冪等: ohlcv_bars は ON CONFLICT (symbol, time) DO UPDATE（ingestion と同じ INSERT_SQL）。
ingestion 稼働中でも安全（現バーは双方が UPSERT）。

実行:
    $env:AEVUM_DB_DSN = "postgresql://postgres:aevum@localhost:5432/aevum"
    .\.venv\Scripts\python.exe scripts/backfill_candles.py --days 7
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root を import 可に

from data.ingestion import (  # noqa: E402
    CANDLE_INTERVAL,
    INSERT_SQL,
    SYMBOLS,
    _fnum,
    _ms_to_dt,
    resolve_dsn,
)

# 各足の長さ(ms)。チャンク境界の前進に使う。
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}
MAX_PER_REQ = 4000  # candleSnapshot の1回上限(~5000)を安全側で回避


def _candle_row(coin: str, c: dict) -> tuple:
    """REST candle dict → ohlcv_bars 行（parse_candle と同じ列順）。"""
    return (
        coin,
        _ms_to_dt(c["t"]),
        _fnum(c["o"]), _fnum(c["h"]), _fnum(c["l"]), _fnum(c["c"]),
        _fnum(c["v"]),
        int(c["n"]) if c.get("n") is not None else None,
        None,  # vwap 未提供
    )


def _fetch(info, coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """[start_ms, end_ms) を MAX_PER_REQ ごとにチャンク取得して連結（同期 SDK）。"""
    step = INTERVAL_MS[interval]
    out: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        chunk_end = min(cur + MAX_PER_REQ * step, end_ms)
        rows = info.candles_snapshot(coin, interval, cur, chunk_end)
        if not rows:
            cur = chunk_end
            continue
        out.extend(rows)
        nxt = int(rows[-1]["t"]) + step
        if nxt <= cur:  # 前進しないなら打ち切り（無限ループ防止）
            break
        cur = nxt
    return out


async def main(days: float, interval: str, symbols: tuple[str, ...]) -> None:
    if interval not in INTERVAL_MS:
        raise SystemExit(f"unsupported interval: {interval} (have {list(INTERVAL_MS)})")

    from hyperliquid.info import Info
    from hyperliquid.utils import constants

    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(days * 86_400_000)

    pool = await asyncpg.create_pool(resolve_dsn(), min_size=1, max_size=4)
    try:
        for coin in symbols:
            raw = _fetch(info, coin, interval, start_ms, now_ms)
            dedup = {int(c["t"]): c for c in raw}  # チャンク重複を t で除去
            rows = sorted((_candle_row(coin, c) for c in dedup.values()), key=lambda r: r[1])
            if rows:
                await pool.executemany(INSERT_SQL["ohlcv_bars"], rows)
            span = f"{rows[0][1]} .. {rows[-1][1]}" if rows else "(none)"
            print(f"{coin}: fetched {len(raw)} -> upserted {len(rows)} bars  {span}")
    finally:
        await pool.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0, help="遡る日数")
    ap.add_argument("--interval", default=CANDLE_INTERVAL, help="足(既定 5m)")
    ap.add_argument("--symbols", default=",".join(SYMBOLS), help="カンマ区切り銘柄")
    args = ap.parse_args()
    syms = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    asyncio.run(main(args.days, args.interval, syms))
