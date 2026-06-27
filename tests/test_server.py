"""api/server.py のテスト（TestClient。fastapi/httpx 必須なので無ければ skip）。"""
from __future__ import annotations

import datetime as dt

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.server import (  # noqa: E402
    HORIZON,
    PRIMARY_SYMBOL,
    PROB_THRESHOLD,
    StateProvider,
    build_snapshot,
    create_app,
)
from shared import volatility  # noqa: E402


class FakeProvider(StateProvider):
    async def predictions(self, symbol=None, limit=20):
        rows = [
            {"symbol": "BTC", "time": "2026-06-26T00:00:00+00:00", "signal": 1,
             "prob_down": 0.1, "prob_flat": 0.2, "prob_up": 0.7, "sigma": 0.01},
            {"symbol": "ETH", "time": "2026-06-26T00:00:00+00:00", "signal": -1,
             "prob_down": 0.6, "prob_flat": 0.3, "prob_up": 0.1, "sigma": 0.02},
        ]
        if symbol:
            rows = [r for r in rows if r["symbol"] == symbol]
        return rows[:limit]

    async def positions(self):
        return [{"symbol": "BTC", "size": 0.5, "entry_price": 60000.0}]

    async def orders(self, limit=50):
        return [{"symbol": "BTC", "side": "buy", "intent": "entry",
                 "status": "open", "risk_passed": True}][:limit]


@pytest.fixture()
def client():
    return TestClient(create_app(FakeProvider(), ws_interval=0.01))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predictions(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert {p["symbol"] for p in r.json()} == {"BTC", "ETH"}


def test_predictions_filter_symbol(client):
    r = client.get("/predictions", params={"symbol": "BTC"})
    body = r.json()
    assert len(body) == 1 and body[0]["symbol"] == "BTC"


def test_positions(client):
    r = client.get("/positions")
    assert r.status_code == 200 and r.json()[0]["symbol"] == "BTC"


def test_orders(client):
    r = client.get("/orders")
    body = r.json()
    assert body[0]["risk_passed"] is True


def test_no_control_endpoints(client):
    # 読み取り専用: 注文を動かす経路が無いこと
    assert client.post("/orders").status_code in (404, 405)
    assert client.post("/kill-switch").status_code == 404


def test_websocket_snapshot(client):
    with client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert "predictions" in snap and "positions" in snap
        assert snap["positions"][0]["symbol"] == "BTC"


# --------------------------------------------------------------------------- #
# build_snapshot（純粋関数）
# --------------------------------------------------------------------------- #
_PRED = {
    "symbol": "BTC", "time": "2026-06-26T09:40:00+00:00", "model_version": "v1",
    "prob_down": 0.1, "prob_flat": 0.2, "prob_up": 0.7, "signal": 1, "sigma": 0.004,
}


def test_build_snapshot_shape_and_sigma():
    now = dt.datetime(2026, 6, 26, 9, 40, tzinfo=dt.timezone.utc)
    entry = dt.datetime(2026, 6, 26, 9, 10, tzinfo=dt.timezone.utc)  # 30分 = 6バー
    snap = build_snapshot(
        now=now, prediction=_PRED, candles=[],
        barriers={"entry": 64000.0, "takeProfit": 66000.0, "stopLoss": 63000.0},
        feature_row={"sigma_ewma": 0.004, "rsi_14": 55.0, "obi_l5": None},
        positions=[{"symbol": "BTC", "unrealized_pnl": 12.5}],
        sl_levels={"BTC": 63000.0}, entry_times={"BTC": entry},
    )
    # equity / winRate は当面 null
    assert snap["metrics"]["equity"] is None
    assert snap["metrics"]["winRate"] is None
    # σ.scaled は shared.volatility.scale_to_horizon と一致（server 自前計算でない）
    assert snap["metrics"]["sigma"]["perBar"] == 0.004
    assert snap["metrics"]["sigma"]["horizon"] == HORIZON
    assert snap["metrics"]["sigma"]["scaled"] == volatility.scale_to_horizon(0.004, HORIZON)
    assert snap["metrics"]["openPnl"] == 12.5
    assert snap["probThreshold"] == PROB_THRESHOLD
    assert snap["chart"]["symbol"] == PRIMARY_SYMBOL
    # 建玉に表示付帯が結合される
    pv = snap["positions"][0]
    assert pv["slLevel"] == 63000.0 and pv["horizon"] == HORIZON and pv["barsHeld"] == 6
    # 特徴量は present なキーのみ（None は除外）
    names = {f["name"] for f in snap["features"]}
    assert "sigma_ewma" in names and "rsi_14" in names and "obi_l5" not in names


def test_build_snapshot_handles_missing():
    now = dt.datetime(2026, 6, 26, 9, 40, tzinfo=dt.timezone.utc)
    snap = build_snapshot(
        now=now, prediction=None, candles=[],
        barriers={"entry": None, "takeProfit": None, "stopLoss": None},
        feature_row=None, positions=[], sl_levels={}, entry_times={},
    )
    assert snap["prediction"] is None
    assert snap["metrics"]["sigma"]["perBar"] is None
    assert snap["metrics"]["sigma"]["scaled"] is None
    assert snap["metrics"]["openPnl"] == 0.0
    assert snap["features"] == [] and snap["positions"] == []


def test_snapshot_endpoint(client):
    # FakeProvider は base snapshot()（{predictions, positions}）。/snapshot も同形を返す。
    r = client.get("/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body and "positions" in body
