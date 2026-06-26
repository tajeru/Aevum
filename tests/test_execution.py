"""live/execution.py の単体テスト。

純粋コア（signal/action/sizing/bracket/SDK写像）と、全注文が risk ゲートを
必ず通る（拒否注文は取引所に届かない）ことを検証する。
"""
from __future__ import annotations

import asyncio
import math

import pytest

from live.execution import (
    ExecConfig,
    bracket_levels,
    decide_action,
    execute_plan,
    plan,
    position_size,
    signal_from_probs,
    submit,
    to_sdk_order,
)
from live.risk import Order, RiskConfig, RiskState

CFG = ExecConfig()


def _state(**kw):
    base = dict(equity=100_000.0, positions={}, mark_prices={"BTC": 60_000.0})
    base.update(kw)
    return RiskState(**base)


class FakeExchange:
    def __init__(self):
        self.calls = []  # ("order"|"market_close", kwargs)

    def order(self, **kw):
        self.calls.append(("order", kw))
        return {"status": "ok", "oid": len(self.calls)}

    def market_close(self, **kw):
        self.calls.append(("market_close", kw))
        return {"status": "ok", "closed": True}


# --------------------------------------------------------------------------- #
# signal / action
# --------------------------------------------------------------------------- #
def test_signal_from_probs():
    assert signal_from_probs(0.1, 0.2, 0.7, 0.5) == 1
    assert signal_from_probs(0.7, 0.2, 0.1, 0.5) == -1
    assert signal_from_probs(0.3, 0.4, 0.3, 0.5) == 0   # 閾値未満 → 様子見
    assert signal_from_probs(0.4, 0.2, 0.4, 0.5) == 0


@pytest.mark.parametrize("sig,pos,expected", [
    (1, 0, "enter_long"), (-1, 0, "enter_short"),
    (1, 1, "hold"), (-1, -1, "hold"),
    (1, -1, "reverse_long"), (-1, 1, "reverse_short"),
    (0, 1, "hold"), (0, 0, "hold"),
])
def test_decide_action(sig, pos, expected):
    assert decide_action(sig, pos) == expected


# --------------------------------------------------------------------------- #
# sizing / bracket
# --------------------------------------------------------------------------- #
def test_position_size_fixed_risk():
    w = 0.01 * math.sqrt(4)
    sl_dist = 100 * (1 - math.exp(-1.0 * w))
    expected = (10_000 * 0.01) / sl_dist
    got = position_size(100.0, 0.01, 10_000.0, sl_mult=1.0, horizon=4, risk_frac=0.01)
    assert got == pytest.approx(expected)


def test_position_size_zero_on_bad_inputs():
    assert position_size(0.0, 0.01, 1e4, sl_mult=1, horizon=4, risk_frac=0.01) == 0.0
    assert position_size(100.0, 0.0, 1e4, sl_mult=1, horizon=4, risk_frac=0.01) == 0.0


def test_bracket_levels_long_and_short():
    w = 0.01 * math.sqrt(4)
    tp, sl = bracket_levels(100.0, "buy", 0.01, pt_mult=1.0, sl_mult=1.0, horizon=4)
    assert tp == pytest.approx(100 * math.exp(w)) and sl == pytest.approx(100 * math.exp(-w))
    assert sl < 100 < tp
    tp_s, sl_s = bracket_levels(100.0, "sell", 0.01, pt_mult=1.0, sl_mult=1.0, horizon=4)
    assert tp_s < 100 < sl_s


# --------------------------------------------------------------------------- #
# SDK 写像
# --------------------------------------------------------------------------- #
def test_to_sdk_entry_limit_maker():
    k = to_sdk_order(Order("BTC", "buy", "entry", "limit", 0.1, 60_000.0))
    assert k["order_type"] == {"limit": {"tif": "Alo"}}
    assert k["reduce_only"] is False and k["is_buy"] is True and k["limit_px"] == 60_000.0


def test_to_sdk_take_profit_reduce_only():
    k = to_sdk_order(Order("BTC", "sell", "take_profit", "limit", 0.1, 60_500.0))
    assert k["order_type"] == {"limit": {"tif": "Alo"}} and k["reduce_only"] is True


def test_to_sdk_stop_loss_trigger():
    k = to_sdk_order(Order("BTC", "sell", "stop_loss", "market", 0.1, 59_000.0))
    assert k["order_type"] == {"trigger": {"triggerPx": 59_000.0, "isMarket": True, "tpsl": "sl"}}
    assert k["reduce_only"] is True and k["limit_px"] == 59_000.0


def test_to_sdk_requires_price():
    with pytest.raises(ValueError):
        to_sdk_order(Order("BTC", "sell", "stop_loss", "market", 0.1, None))
    with pytest.raises(ValueError):
        to_sdk_order(Order("BTC", "buy", "entry", "limit", 0.1, None))


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def _pred(symbol="BTC", down=0.1, flat=0.2, up=0.7, sigma=0.01):
    return {"symbol": symbol, "prob_down": down, "prob_flat": flat, "prob_up": up, "sigma": sigma}


def test_plan_enter_long_with_brackets():
    p = plan(_pred(), _state(), CFG)
    assert p.close is None
    assert p.entry.side == "buy" and p.entry.intent == "entry"
    assert [b.intent for b in p.brackets] == ["take_profit", "stop_loss"]
    assert p.brackets[0].side == "sell" and p.brackets[1].side == "sell"


def test_plan_reverse_closes_then_enters():
    st = _state(positions={"BTC": 0.5})           # ロング保有
    p = plan(_pred(down=0.7, flat=0.2, up=0.1), st, CFG)  # 売りシグナル → ドテン
    assert p.close is not None and p.close.side == "sell" and p.close.size == pytest.approx(0.5)
    assert p.entry.side == "sell" and p.entry.intent == "entry"


def test_plan_hold_when_flat_signal():
    p = plan(_pred(down=0.3, flat=0.4, up=0.3), _state(), CFG)
    assert p.close is None and p.entry is None and p.brackets == []


def test_plan_empty_without_mark():
    p = plan(_pred(), _state(mark_prices={}), CFG)
    assert p.entry is None


# --------------------------------------------------------------------------- #
# ゲート付き発注（全注文がゲートを通る）
# --------------------------------------------------------------------------- #
def test_submit_approved_sends_and_records():
    ex = FakeExchange()
    order = Order("BTC", "buy", "entry", "limit", 0.1, 60_000.0)
    rec = asyncio.run(submit(order, _state(), CFG, ex))
    assert rec["risk_passed"] is True and rec["status"] == "open"
    assert len(ex.calls) == 1  # 取引所に届いた


def test_submit_rejected_never_reaches_exchange():
    ex = FakeExchange()
    # max_order_size 超過 → ゲート拒否
    order = Order("BTC", "buy", "entry", "limit", CFG.risk.max_order_size + 1, 60_000.0)
    rec = asyncio.run(submit(order, _state(), CFG, ex))
    assert rec["risk_passed"] is False and rec["status"] == "rejected"
    assert ex.calls == []  # 送信されない


def test_submit_kill_switch_blocks_entry_but_not_exit():
    ex = FakeExchange()
    st = _state(kill_switch=True, positions={"BTC": 0.5})
    entry = Order("BTC", "buy", "entry", "limit", 0.1, 60_000.0)
    sl = Order("BTC", "sell", "stop_loss", "market", 0.5, 59_000.0)
    r_entry = asyncio.run(submit(entry, st, CFG, ex))
    r_exit = asyncio.run(submit(sl, st, CFG, ex))
    assert r_entry["risk_passed"] is False and r_exit["risk_passed"] is True
    assert len(ex.calls) == 1  # 退避のみ送信


def test_execute_plan_close_then_entry():
    ex = FakeExchange()
    st = _state(positions={"BTC": 0.5})
    p = plan(_pred(down=0.7, flat=0.2, up=0.1), st, CFG)
    recs = asyncio.run(execute_plan(p, st, CFG, ex))
    assert len(recs) == 2  # close + entry
    assert ex.calls[0][0] == "market_close"          # 先に即時クローズ
    assert ex.calls[1][0] == "order" and ex.calls[1][1]["reduce_only"] is False  # 反対へエントリー
