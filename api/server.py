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
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from live.execution import ExecConfig  # prob_threshold / horizon の唯一ソース
from shared import volatility  # σ スケールの唯一定義（scale_to_horizon）
from shared.feature_names import SIGMA_FEATURE  # bar_features の σ 列名


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


# --------------------------------------------------------------------------- #
# スナップショット組み立て（純粋関数・I/O なし）
#   取得済みデータを UI の DashboardState 形へ整形するだけ。ロジックは持たない。
# --------------------------------------------------------------------------- #
PRIMARY_SYMBOL = "BTC"   # チャート/シグナルの主表示銘柄
BAR_SECONDS = 300        # 5 分足（barsHeld 換算用）
CANDLE_LIMIT = 200       # チャートへ返す直近バー数

_EXEC = ExecConfig()                 # prob_threshold / horizon の唯一ソース
HORIZON = _EXEC.horizon              # 縦バリア（= DEFAULT_HORIZON = 48）
PROB_THRESHOLD = _EXEC.prob_threshold

# SignalPanel に出すライブ特徴量（bar_features 列名から代表を抜粋）
LIVE_FEATURE_KEYS = (SIGMA_FEATURE, "obi_l5", "rsi_14", "macd_hist", "spread_bps", "funding_rate")

# --------------------------------------------------------------------------- #
# Ingestion monitoring constants (cadence-based stale thresholds)
#   ohlcv_bars          : 5-min candle bars -> new row every ~300s per symbol
#   orderbook_snapshots : real-time l2Book  -> should refresh in seconds
#   funding_oi          : activeAssetCtx    -> ~60s cadence
# --------------------------------------------------------------------------- #
_MONITORING_TABLES: tuple[str, ...] = ("ohlcv_bars", "orderbook_snapshots", "funding_oi")

MONITORING_STALE_SECONDS: dict[str, int] = {
    "ohlcv_bars": 7 * 60,        # 7 min  (cadence: 5 min)
    "orderbook_snapshots": 60,   # 60 s   (cadence: real-time)
    "funding_oi": 3 * 60,        # 3 min  (cadence: ~1 min)
}

_MONITORING_STATS_SQL = """\
SELECT
    symbol,
    MAX(time)                                                          AS last_write_at,
    EXTRACT(EPOCH FROM (NOW() - MAX(time)))::BIGINT                   AS seconds_since_last_write,
    COUNT(*) FILTER (WHERE time > NOW() - INTERVAL '1 hour')          AS rows_last_1h,
    COUNT(*)                                                           AS rows_total,
    MIN(time)                                                          AS oldest_at,
    EXTRACT(EPOCH FROM (MAX(time) - MIN(time)))::BIGINT               AS span_seconds
FROM {table}
GROUP BY symbol
"""

_MONITORING_EMPTY: dict = {
    "last_write_at": None,
    "seconds_since_last_write": None,
    "rows_last_1h": 0,
    "rows_total": 0,
    "oldest_at": None,
    "span_seconds": None,
}


def _bars_held(entry_time, now, *, bar_seconds: int = BAR_SECONDS) -> int:
    """エントリー発注時刻からの経過バー数（発注時刻近似・スキーマ変更なし）。"""
    if entry_time is None:
        return 0
    secs = (now - entry_time).total_seconds()
    return max(0, int(secs // bar_seconds))


def _sigma_view(per_bar: Optional[float], horizon: int = HORIZON) -> dict:
    """σ 表示: per-bar と ×√horizon。スケールは shared.volatility.scale_to_horizon のみ
    （server 内で σ×√horizon を自前計算しない＝σ 単一定義の遵守。呼び出し側スケールの一例）。"""
    if per_bar is None:
        return {"perBar": None, "horizon": horizon, "scaled": None}
    return {"perBar": per_bar, "horizon": horizon,
            "scaled": volatility.scale_to_horizon(per_bar, horizon)}


def build_snapshot(*, now, prediction, candles, barriers, feature_row,
                   positions, sl_levels, entry_times) -> dict:
    """取得済みデータを DashboardState 形へ（connected/latencyMs はクライアント側で付与）。

    equity / winRate は当面 null（DB に永続化テーブルが無く winRate 定義が未確定。
    UI 側でレイアウト枠は維持）。純粋な整形のみでロジックを持たない。
    """
    per_bar = prediction.get("sigma") if prediction else None

    open_pnl = 0.0
    if positions:
        open_pnl = float(sum((p.get("unrealized_pnl") or 0.0) for p in positions))

    pos_views = []
    for p in positions:
        sym = p.get("symbol")
        pos_views.append({
            **p,
            "slLevel": sl_levels.get(sym),
            "barsHeld": _bars_held(entry_times.get(sym), now),
            "horizon": HORIZON,
        })

    features = []
    if feature_row:
        for k in LIVE_FEATURE_KEYS:
            v = feature_row.get(k)
            if v is not None:
                features.append({"name": k, "value": float(v)})

    return {
        "updatedAt": now.isoformat(),
        "metrics": {
            "equity": None,        # DB に永続化ソース無し（別途方針決定）
            "openPnl": open_pnl,
            "winRate": None,       # 定義未確定（別途方針決定）
            "sigma": _sigma_view(per_bar),
        },
        "chart": {"symbol": PRIMARY_SYMBOL, "candles": candles, "barriers": barriers},
        "prediction": prediction,
        "probThreshold": PROB_THRESHOLD,
        "features": features,
        "positions": pos_views,
    }


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

    async def ingestion_status(self) -> dict:
        raise NotImplementedError


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

    # --- snapshot 用の細粒度クエリ（read-only） --- #
    async def _latest_prediction(self, symbol: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT * FROM model_predictions WHERE symbol = $1 ORDER BY time DESC LIMIT 1",
            symbol,
        )
        return _jsonable(row) if row else None

    async def _candles(self, symbol: str, limit: int) -> list[dict]:
        """ohlcv_bars 直近 limit 本を昇順・エポック秒で返す（lightweight-charts 用）。"""
        rows = await self.pool.fetch(
            "SELECT time, open, high, low, close FROM ohlcv_bars "
            "WHERE symbol = $1 ORDER BY time DESC LIMIT $2",
            symbol, limit,
        )
        return [
            {"time": int(r["time"].timestamp()), "open": r["open"],
             "high": r["high"], "low": r["low"], "close": r["close"]}
            for r in reversed(rows)
        ]

    async def _latest_order_price(self, symbol: str, intent: str) -> Optional[float]:
        """銘柄×intent の最新ブラケット価格。stop_loss は待機ストップに price=triggerPx が
        入る（即時成行クローズは price NULL なので除外）。entry/take_profit も同様に price。"""
        row = await self.pool.fetchrow(
            "SELECT price FROM orders WHERE symbol = $1 AND intent = $2 "
            "AND price IS NOT NULL ORDER BY time DESC LIMIT 1",
            symbol, intent,
        )
        return float(row["price"]) if row and row["price"] is not None else None

    async def _entry_time(self, symbol: str):
        """現建玉のオープン時刻近似 = 最新 entry 注文の発注時刻（スキーマ変更なし）。"""
        row = await self.pool.fetchrow(
            "SELECT time FROM orders WHERE symbol = $1 AND intent = 'entry' "
            "ORDER BY time DESC LIMIT 1",
            symbol,
        )
        return row["time"] if row else None

    async def _latest_feature_row(self, symbol: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT * FROM bar_features WHERE symbol = $1 ORDER BY time DESC LIMIT 1",
            symbol,
        )
        return dict(row) if row else None

    async def snapshot(self) -> dict:
        """DashboardState 形の集約スナップショット（read-only）。整形は build_snapshot に委譲。"""
        now = datetime.now(timezone.utc)
        sym = PRIMARY_SYMBOL

        prediction = await self._latest_prediction(sym)
        candles = await self._candles(sym, CANDLE_LIMIT)
        barriers = {
            "entry": await self._latest_order_price(sym, "entry"),
            "takeProfit": await self._latest_order_price(sym, "take_profit"),
            "stopLoss": await self._latest_order_price(sym, "stop_loss"),
        }
        feature_row = await self._latest_feature_row(sym)
        positions = await self.positions()

        sl_levels: dict = {}
        entry_times: dict = {}
        for p in positions:
            s = p["symbol"]
            sl_levels[s] = await self._latest_order_price(s, "stop_loss")
            entry_times[s] = await self._entry_time(s)

        return build_snapshot(
            now=now, prediction=prediction, candles=candles, barriers=barriers,
            feature_row=feature_row, positions=positions,
            sl_levels=sl_levels, entry_times=entry_times,
        )

    async def ingestion_status(self) -> dict:
        """Return freshness / throughput / accumulation stats for the 3 raw ingestion tables."""
        tables: dict = {}
        for table in _MONITORING_TABLES:
            rows = await self.pool.fetch(_MONITORING_STATS_SQL.format(table=table))
            by_sym = {r["symbol"]: _jsonable(r) for r in rows}
            tables[table] = {
                sym: by_sym.get(sym, dict(_MONITORING_EMPTY))
                for sym in ("BTC", "ETH")
            }
        return {"tables": tables}


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

    @app.get("/snapshot")
    async def snapshot():
        return await app.state.provider.snapshot()

    @app.get("/monitoring/ingestion")
    async def monitoring_ingestion():
        return await app.state.provider.ingestion_status()

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
