"""live/risk.py — リスク管理ゲート（全注文が必ず通る単一の関門）.

CLAUDE.md 不変条件:
* すべての注文は `check_order` を通す。UI 由来でも迂回不可。
* このモジュールは純粋関数（隠れた I/O なし）。現在状態は呼び出し側(execution.py)が
  RiskState として渡し、結果(approved)を `orders.risk_passed` に記録する。

確定仕様（user-confirmed）
-------------------------
* 上限超過のエントリーは **ハードリジェクト**（サイズを勝手に縮めない）。
* kill-switch / 日次損失制限の発動時は **新規エントリーのみ** ブロック。
  退避（take_profit / stop_loss = ポジション縮小・決済）は常に許可（必ず閉じられる）。
* 上限は想定元本(USD)基準: 銘柄別ポジション・合計エクスポージャ・レバレッジ。
* 発注の非対称設計（entry/take_profit=limit, stop_loss=market）もゲートで検証。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_SIDES = ("buy", "sell")
_INTENTS = ("entry", "take_profit", "stop_loss")


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str                     # 'buy' | 'sell'
    intent: str                   # 'entry' | 'take_profit' | 'stop_loss'
    order_type: str               # 'limit' | 'market'
    size: float                   # 契約数（> 0）
    price: Optional[float] = None  # 指値価格。成行は None


@dataclass
class RiskState:
    equity: float                                 # 口座資産(USD)
    positions: dict[str, float] = field(default_factory=dict)     # symbol -> 符号付き建玉(契約数, +買/-売)
    mark_prices: dict[str, float] = field(default_factory=dict)   # symbol -> mark 価格
    daily_pnl: float = 0.0                         # 当日 実現+含み 損益(USD)
    open_orders: int = 0                           # 現在の未約定注文数
    kill_switch: bool = False                      # 手動グローバル停止


@dataclass(frozen=True)
class RiskConfig:
    max_position_notional: float = 50_000.0   # 銘柄別 最大想定元本(USD)
    max_total_notional: float = 100_000.0     # 合計 最大エクスポージャ(USD)
    max_leverage: float = 5.0                 # 合計想定元本 / equity
    min_order_size: float = 1e-4              # 最小サイズ(契約)
    max_order_size: float = 100.0            # 最大サイズ(契約)
    max_price_deviation: float = 0.05         # 指値の mark からの乖離上限(5%)
    max_open_orders: int = 20
    daily_loss_limit: float = 1_000.0         # daily_pnl <= -limit でエントリー停止(USD)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = "ok"


def _reject(reason: str) -> RiskDecision:
    return RiskDecision(False, reason)


def _valid_intent_type(order: Order) -> bool:
    """非対称設計: entry/take_profit=limit, stop_loss=market。"""
    if order.intent in ("entry", "take_profit"):
        return order.order_type == "limit"
    if order.intent == "stop_loss":
        return order.order_type == "market"
    return False


def check_order(order: Order, state: RiskState, config: RiskConfig = RiskConfig()) -> RiskDecision:
    """注文をリスクゲートに通す。approve/reject と理由を返す（純粋関数）。"""
    # --- 全注文に課す基本妥当性 ---
    if order.side not in _SIDES:
        return _reject(f"invalid side: {order.side!r}")
    if order.intent not in _INTENTS:
        return _reject(f"invalid intent: {order.intent!r}")
    if not (order.size > 0):
        return _reject("size must be > 0")
    if order.size < config.min_order_size:
        return _reject(f"size {order.size} < min {config.min_order_size}")
    if order.size > config.max_order_size:
        return _reject(f"size {order.size} > max {config.max_order_size}")
    if not _valid_intent_type(order):
        return _reject(f"intent/order_type violates asymmetric design: {order.intent}/{order.order_type}")

    mark = state.mark_prices.get(order.symbol)

    # 価格サニティ（指値のみ。成行=stop_loss は対象外で約定保証）
    if order.order_type == "limit":
        if order.price is None or order.price <= 0:
            return _reject("limit order requires a positive price")
        if mark is not None and mark > 0:
            dev = abs(order.price - mark) / mark
            if dev > config.max_price_deviation:
                return _reject(f"price deviation {dev:.4f} > {config.max_price_deviation}")

    # 退避（縮小/決済）は基本妥当性のみで常に許可（必ず閉じられる）
    if order.intent != "entry":
        return RiskDecision(True, "ok (exit allowed)")

    # --- エントリー（エクスポージャ増加）のみのチェック ---
    if state.kill_switch:
        return _reject("kill switch active")
    if state.daily_pnl <= -config.daily_loss_limit:
        return _reject(f"daily loss limit hit (pnl={state.daily_pnl})")
    if state.open_orders >= config.max_open_orders:
        return _reject(f"max open orders {config.max_open_orders} reached")
    if mark is None or mark <= 0:
        return _reject(f"no mark price for {order.symbol}")
    if not (state.equity > 0):
        return _reject("non-positive equity")

    signed = order.size if order.side == "buy" else -order.size
    cur = state.positions.get(order.symbol, 0.0)
    new_sym_notional = abs(cur + signed) * mark
    if new_sym_notional > config.max_position_notional:
        return _reject(f"position notional {new_sym_notional:.0f} > {config.max_position_notional:.0f}")

    # 合計エクスポージャ（当該銘柄の寄与を新値に置換）
    total = 0.0
    for sym, pos in state.positions.items():
        m = state.mark_prices.get(sym)
        if m:
            total += abs(pos) * m
    total = total - abs(cur) * mark + new_sym_notional
    if total > config.max_total_notional:
        return _reject(f"total notional {total:.0f} > {config.max_total_notional:.0f}")

    leverage = total / state.equity
    if leverage > config.max_leverage:
        return _reject(f"leverage {leverage:.2f} > {config.max_leverage}")

    return RiskDecision(True, "ok")
