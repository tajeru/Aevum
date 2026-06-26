"""live/risk.py のリスクゲート単体テスト。

全注文が通る単一の関門。各チェック（基本妥当性・非対称設計・価格サニティ・
エントリーのエクスポージャ/レバレッジ/kill-switch/損失制限、退避は常時許可）を網羅。
"""
from __future__ import annotations

import pytest

from live.risk import Order, RiskConfig, RiskState, check_order

CFG = RiskConfig()


def _state(**kw) -> RiskState:
    base = dict(equity=100_000.0, positions={}, mark_prices={"BTC": 60_000.0, "ETH": 3_000.0})
    base.update(kw)
    return RiskState(**base)


def _entry(symbol="BTC", side="buy", size=0.1, price=60_000.0):
    return Order(symbol, side, "entry", "limit", size, price)


# --------------------------------------------------------------------------- #
# 正常系
# --------------------------------------------------------------------------- #
def test_valid_entry_approved():
    d = check_order(_entry(), _state(), CFG)
    assert d.approved


def test_valid_stop_loss_market_approved():
    d = check_order(Order("BTC", "sell", "stop_loss", "market", 0.1), _state(), CFG)
    assert d.approved


# --------------------------------------------------------------------------- #
# 基本妥当性
# --------------------------------------------------------------------------- #
def test_invalid_side():
    assert not check_order(_entry(side="hold"), _state(), CFG).approved


def test_invalid_intent():
    assert not check_order(Order("BTC", "buy", "scalp", "limit", 0.1, 60_000.0), _state(), CFG).approved


@pytest.mark.parametrize("size", [0.0, -1.0])
def test_nonpositive_size(size):
    assert not check_order(_entry(size=size), _state(), CFG).approved


def test_size_below_min_and_above_max():
    assert not check_order(_entry(size=1e-9), _state(), CFG).approved
    assert not check_order(_entry(size=CFG.max_order_size + 1), _state(), CFG).approved


# --------------------------------------------------------------------------- #
# 非対称設計
# --------------------------------------------------------------------------- #
def test_entry_must_be_limit():
    assert not check_order(Order("BTC", "buy", "entry", "market", 0.1), _state(), CFG).approved


def test_stop_loss_must_be_market():
    assert not check_order(Order("BTC", "sell", "stop_loss", "limit", 0.1, 60_000.0), _state(), CFG).approved


def test_take_profit_must_be_limit():
    assert check_order(Order("BTC", "sell", "take_profit", "limit", 0.1, 60_500.0), _state(), CFG).approved
    assert not check_order(Order("BTC", "sell", "take_profit", "market", 0.1), _state(), CFG).approved


# --------------------------------------------------------------------------- #
# 価格サニティ
# --------------------------------------------------------------------------- #
def test_limit_requires_price():
    assert not check_order(Order("BTC", "buy", "entry", "limit", 0.1, None), _state(), CFG).approved


def test_price_deviation_rejected():
    # mark=60000, 10% 乖離 → 上限5% 超
    assert not check_order(_entry(price=66_000.0), _state(), CFG).approved


def test_price_within_deviation_ok():
    assert check_order(_entry(price=61_000.0), _state(), CFG).approved  # ~1.7%


# --------------------------------------------------------------------------- #
# 退避は常時許可（kill-switch / 損失制限下でも）
# --------------------------------------------------------------------------- #
def test_exit_allowed_under_kill_switch():
    st = _state(kill_switch=True, positions={"BTC": 0.5})
    assert check_order(Order("BTC", "sell", "stop_loss", "market", 0.5), st, CFG).approved
    assert check_order(Order("BTC", "sell", "take_profit", "limit", 0.5, 60_500.0), st, CFG).approved


def test_exit_allowed_under_daily_loss():
    st = _state(daily_pnl=-CFG.daily_loss_limit - 1, positions={"BTC": 0.5})
    assert check_order(Order("BTC", "sell", "stop_loss", "market", 0.5), st, CFG).approved


# --------------------------------------------------------------------------- #
# エントリーのみのチェック
# --------------------------------------------------------------------------- #
def test_kill_switch_blocks_entry():
    assert not check_order(_entry(), _state(kill_switch=True), CFG).approved


def test_daily_loss_blocks_entry():
    assert not check_order(_entry(), _state(daily_pnl=-CFG.daily_loss_limit), CFG).approved


def test_max_open_orders_blocks_entry():
    assert not check_order(_entry(), _state(open_orders=CFG.max_open_orders), CFG).approved


def test_no_mark_price_blocks_entry():
    assert not check_order(_entry(symbol="BTC"), _state(mark_prices={}), CFG).approved


def test_nonpositive_equity_blocks_entry():
    assert not check_order(_entry(), _state(equity=0.0), CFG).approved


# --------------------------------------------------------------------------- #
# 上限超過はハードリジェクト
# --------------------------------------------------------------------------- #
def test_position_notional_limit():
    # 1.0 BTC * 60000 = 60000 > 50000 上限
    assert not check_order(_entry(size=1.0), _state(), CFG).approved


def test_total_notional_limit():
    # 既存 ETH 想定元本 90000 + 新規 BTC 0.5*60000=30000 = 120000 > 100000
    st = _state(positions={"ETH": 30.0})  # 30*3000=90000
    assert not check_order(_entry(size=0.5), st, CFG).approved


def test_leverage_limit():
    cfg = RiskConfig(max_leverage=1.0, max_position_notional=1e12, max_total_notional=1e12)
    # 0.5*60000=30000 想定元本, equity 10000 → leverage 3 > 1
    assert not check_order(_entry(size=0.5), _state(equity=10_000.0), cfg).approved


def test_entry_reducing_position_passes():
    # 既に売り建て -0.5、buy entry 0.1 は建玉を縮小 → エクスポージャ減 → 許可
    st = _state(positions={"BTC": -0.5})
    assert check_order(_entry(side="buy", size=0.1), st, CFG).approved
