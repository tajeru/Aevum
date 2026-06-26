"""api/server.py — FastAPI + WebSocket（状態配信のみ・ロジックなし）.

CLAUDE.md 不変条件:
* server.py は状態配信のみ。ロジックを持たない。
* UI 由来のコマンドもリスクゲートを迂回できない（現状は読み取り専用＝注文経路なし）。

設計:
* 状態取得は StateProvider 抽象に委譲（DB を読むだけ）。create_app(provider) に注入し、
  TestClient で DB なしにテストできる。本番は create_db_app(dsn) が asyncpg で背後を埋める。
* 提供: GET /health, /predictions, /positions, /orders, および WebSocket /ws（定期スナップショット）。

将来 kill-switch を足す場合も、server はロジックを持たず control テーブルのフラグを
切り替えるだけにし、execution が読み RiskState 経由で risk.py が判断する（ゲート迂回禁止）。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from shared.feature_names import SIGMA_FEATURE  # noqa: F401  (将来 σ 配信の参照用)


def _jsonable(row) -> dict:
    """asyncpg Record/dict を JSON 化可能な dict へ（datetime→ISO, Decimal→float）。"""
    out = {}
    for k, v in dict(row).items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, Decimal):
            out[k] = float(v)
        else:
            out[k] = v
    return out


class StateProvider:
    """状態取得インターフェース（DB を読むだけ。ロジックは持たない）。"""

    async def health(self) -> dict:
        return {"status": "ok"}

    async def predictions(self, symbol: Optional[str] = None, limit: int = 20) -> list[dict]:
        raise NotImplementedError

    async def positions(self) -> list[dict]:
        raise NotImplementedError

    async def orders(self, limit: int = 50) -> list[dict]:
        raise NotImplementedError

    async def snapshot(self) -> dict:
        return {
            "predictions": await self.predictions(limit=10),
            "positions": await self.positions(),
        }


class DbStateProvider(StateProvider):
    """asyncpg プール背後の本番プロバイダ。"""

    def __init__(self, pool) -> None:
        self.pool = pool

    async def predictions(self, symbol: Optional[str] = None, limit: int = 20) -> list[dict]:
        if symbol:
            rows = await self.pool.fetch(
                "SELECT * FROM model_predictions WHERE symbol = $1 ORDER BY time DESC LIMIT $2",
                symbol, limit,
            )
        else:
            rows = await self.pool.fetch(
                "SELECT * FROM model_predictions ORDER BY time DESC LIMIT $1", limit
            )
        return [_jsonable(r) for r in rows]

    async def positions(self) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT DISTINCT ON (symbol) * FROM positions ORDER BY symbol, time DESC"
        )
        return [_jsonable(r) for r in rows]

    async def orders(self, limit: int = 50) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT * FROM orders ORDER BY time DESC LIMIT $1", limit
        )
        return [_jsonable(r) for r in rows]


def _register(app: FastAPI, ws_interval: float) -> None:
    @app.get("/health")
    async def health():
        return await app.state.provider.health()

    @app.get("/predictions")
    async def predictions(symbol: Optional[str] = None, limit: int = 20):
        return await app.state.provider.predictions(symbol, limit)

    @app.get("/positions")
    async def positions():
        return await app.state.provider.positions()

    @app.get("/orders")
    async def orders(limit: int = 50):
        return await app.state.provider.orders(limit)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(await app.state.provider.snapshot())
                await asyncio.sleep(ws_interval)
        except WebSocketDisconnect:
            pass


def create_app(provider: StateProvider, *, ws_interval: float = 1.0) -> FastAPI:
    """プロバイダ注入で app を生成（テスト用）。"""
    app = FastAPI(title="Aevum API")
    app.state.provider = provider
    _register(app, ws_interval)
    return app


def create_db_app(dsn: Optional[str] = None, *, ws_interval: float = 1.0) -> FastAPI:
    """本番 app。lifespan で asyncpg プール＋DbStateProvider を用意。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncpg

        from data.ingestion import resolve_dsn

        pool = await asyncpg.create_pool(dsn or resolve_dsn(), min_size=1, max_size=4)
        app.state.provider = DbStateProvider(pool)
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(title="Aevum API", lifespan=lifespan)
    _register(app, ws_interval)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_db_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
