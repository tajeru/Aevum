"""api/server.py のテスト（TestClient。fastapi/httpx 必須なので無ければ skip）。"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.server import StateProvider, create_app  # noqa: E402


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
