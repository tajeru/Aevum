"""data/ingestion.py の単体テスト。

I/O を持たない部分を網羅:
  1. メッセージ解析（candle / l2Book / activeAssetCtx）— 正常・異常・文字列数値
  2. INSERT のプレースホルダ数 == parse_* の返すタプル長（列ズレ防止）
  3. resolve_dsn（AEVUM_DB_DSN / PG* フォールバック）
  4. BatchWriter のバッファ振り分け・バッチ flush・失敗時の挙動（fake pool）
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

from data.ingestion import (
    INSERT_SQL,
    TABLES,
    BatchWriter,
    make_callbacks,
    parse_active_ctx,
    parse_candle,
    parse_l2book,
    resolve_dsn,
)

# 2023-11-14T22:13:20Z
TS_MS = 1_700_000_000_000
TS_DT = datetime.fromtimestamp(TS_MS / 1000.0, tz=timezone.utc)


# --------------------------------------------------------------------------- #
# candle
# --------------------------------------------------------------------------- #
def _candle_msg(coin="BTC", str_nums=True):
    o, h, l, c, v = ("65000.0", "65100.5", "64900.0", "65050.0", "12.5")
    if not str_nums:
        o, h, l, c, v = 65000.0, 65100.5, 64900.0, 65050.0, 12.5
    return {
        "channel": "candle",
        "data": {"t": TS_MS, "T": TS_MS + 300000, "s": coin, "i": "5m",
                 "o": o, "h": h, "l": l, "c": c, "v": v, "n": 42},
    }


def test_parse_candle_string_numbers():
    row = parse_candle(_candle_msg(str_nums=True))
    assert row == ("BTC", TS_DT, 65000.0, 65100.5, 64900.0, 65050.0, 12.5, 42, None)
    assert row[1].tzinfo is not None  # tz-aware


def test_parse_candle_numeric_numbers():
    assert parse_candle(_candle_msg(str_nums=False)) == \
        ("BTC", TS_DT, 65000.0, 65100.5, 64900.0, 65050.0, 12.5, 42, None)


def test_parse_candle_rejects_wrong_channel_and_coin():
    assert parse_candle({"channel": "l2Book", "data": {}}) is None
    assert parse_candle(_candle_msg(coin="SOL")) is None
    assert parse_candle("not a dict") is None
    assert parse_candle({"channel": "candle"}) is None


def test_parse_candle_missing_field_returns_none():
    msg = _candle_msg()
    del msg["data"]["o"]
    assert parse_candle(msg) is None


# --------------------------------------------------------------------------- #
# l2Book
# --------------------------------------------------------------------------- #
def _l2_msg(coin="ETH"):
    return {
        "channel": "l2Book",
        "data": {
            "coin": coin,
            "time": TS_MS,
            "levels": [
                [{"px": "3500.0", "sz": "10.0", "n": 3}, {"px": "3499.5", "sz": "5.0", "n": 2}],
                [{"px": "3500.5", "sz": "8.0", "n": 1}, {"px": "3501.0", "sz": "4.0", "n": 2}],
            ],
        },
    }


def test_parse_l2book_ok():
    row = parse_l2book(_l2_msg())
    assert row == (
        "ETH", TS_DT,
        [3500.0, 3499.5], [10.0, 5.0],
        [3500.5, 3501.0], [8.0, 4.0],
    )


def test_parse_l2book_rejects_bad_shape():
    bad = _l2_msg()
    bad["data"]["levels"] = [[]]  # 2要素でない
    assert parse_l2book(bad) is None
    assert parse_l2book(_l2_msg(coin="DOGE")) is None
    assert parse_l2book({"channel": "candle", "data": {}}) is None


# --------------------------------------------------------------------------- #
# activeAssetCtx
# --------------------------------------------------------------------------- #
def _ctx_msg(coin="BTC", premium="0.0001"):
    ctx = {"funding": "0.0000125", "openInterest": "1234.5",
           "markPx": "65010.0", "oraclePx": "65000.0", "dayNtlVlm": "9e8"}
    if premium is not None:
        ctx["premium"] = premium
    return {"channel": "activeAssetCtx", "data": {"coin": coin, "ctx": ctx}}


def test_parse_active_ctx_ok():
    rt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_active_ctx(_ctx_msg(), rt) == \
        ("BTC", rt, 0.0000125, 1234.5, 65010.0, 65000.0, 0.0001)


def test_parse_active_ctx_premium_optional():
    rt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = parse_active_ctx(_ctx_msg(premium=None), rt)
    assert row[-1] is None  # premium 欠如 → None


def test_parse_active_ctx_rejects():
    rt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert parse_active_ctx({"channel": "activeAssetCtx", "data": {"coin": "BTC"}}, rt) is None
    assert parse_active_ctx(_ctx_msg(coin="XRP"), rt) is None


# --------------------------------------------------------------------------- #
# INSERT プレースホルダ数 == 行タプル長
# --------------------------------------------------------------------------- #
def _placeholder_count(sql: str) -> int:
    return max(int(m) for m in re.findall(r"\$(\d+)", sql))


def test_insert_arity_matches_parsed_rows():
    rt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert set(INSERT_SQL) == set(TABLES)
    assert _placeholder_count(INSERT_SQL["ohlcv_bars"]) == len(parse_candle(_candle_msg()))
    assert _placeholder_count(INSERT_SQL["orderbook_snapshots"]) == len(parse_l2book(_l2_msg()))
    assert _placeholder_count(INSERT_SQL["funding_oi"]) == len(parse_active_ctx(_ctx_msg(), rt))


# --------------------------------------------------------------------------- #
# resolve_dsn
# --------------------------------------------------------------------------- #
def test_resolve_dsn_prefers_explicit():
    assert resolve_dsn({"AEVUM_DB_DSN": "postgresql://x/y"}) == "postgresql://x/y"


def test_resolve_dsn_pg_fallback():
    env = {"PGHOST": "db", "PGPORT": "6000", "PGUSER": "u", "PGPASSWORD": "p", "PGDATABASE": "aevum"}
    assert resolve_dsn(env) == "postgresql://u:p@db:6000/aevum"


def test_resolve_dsn_pg_defaults_no_password():
    assert resolve_dsn({}) == "postgresql://postgres@localhost:5432/aevum"


# --------------------------------------------------------------------------- #
# BatchWriter（fake pool）
# --------------------------------------------------------------------------- #
class _FakePool:
    def __init__(self, fail: bool = False):
        self.calls: list[tuple[str, list]] = []
        self.fail = fail

    async def executemany(self, sql, rows):
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((sql, list(rows)))


def test_batchwriter_groups_and_flushes():
    async def run():
        pool = _FakePool()
        w = BatchWriter(pool)
        w.bind_loop(asyncio.get_running_loop())
        w._append("ohlcv_bars", ("BTC", TS_DT, 1, 2, 3, 4, 5, 6, None))
        w._append("ohlcv_bars", ("ETH", TS_DT, 1, 2, 3, 4, 5, 6, None))
        w._append("orderbook_snapshots", ("BTC", TS_DT, [1.0], [2.0], [3.0], [4.0]))
        await w.flush_once()
        return pool, w

    pool, w = asyncio.run(run())
    by_sql = {sql: rows for sql, rows in pool.calls}
    assert len(by_sql[INSERT_SQL["ohlcv_bars"]]) == 2
    assert len(by_sql[INSERT_SQL["orderbook_snapshots"]]) == 1
    assert "funding_oi" not in [s for s, _ in pool.calls]  # 空テーブルは flush しない
    assert w.written == 3 and w.dropped == 0
    # flush 後はバッファが空
    assert all(len(b) == 0 for b in w._buffers.values())


def test_batchwriter_flush_failure_is_contained():
    async def run():
        pool = _FakePool(fail=True)
        w = BatchWriter(pool)
        w.bind_loop(asyncio.get_running_loop())
        w._append("ohlcv_bars", ("BTC", TS_DT, 1, 2, 3, 4, 5, 6, None))
        await w.flush_once()  # 例外を投げず dropped に計上
        return w

    w = asyncio.run(run())
    assert w.dropped == 1 and w.written == 0


def test_callbacks_route_to_writer():
    async def run():
        pool = _FakePool()
        w = BatchWriter(pool)
        w.bind_loop(asyncio.get_running_loop())
        cb = make_callbacks(w, now_fn=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
        cb["candle"](_candle_msg())
        cb["l2Book"](_l2_msg())
        cb["activeAssetCtx"](_ctx_msg())
        cb["candle"]({"channel": "candle", "data": {"s": "SOL"}})  # 無視される
        await asyncio.sleep(0)  # call_soon_threadsafe を処理
        await w.flush_once()
        return pool

    pool = asyncio.run(run())
    tables_written = {sql for sql, _ in pool.calls}
    assert INSERT_SQL["ohlcv_bars"] in tables_written
    assert INSERT_SQL["orderbook_snapshots"] in tables_written
    assert INSERT_SQL["funding_oi"] in tables_written
    total = sum(len(rows) for _, rows in pool.calls)
    assert total == 3  # SOL は除外
