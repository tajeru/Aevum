"""live/execution.py — Pi側: Hyperliquid 発注（トレーディングループ）.

CLAUDE.md 不変条件:
* すべての注文は `risk.check_order` を通す（UI 由来も迂回不可）。承認注文のみ送信し、
  結果(approved)を orders.risk_passed に記録する。
* ライブのバリア幅 σ は `shared/volatility.py`（labels と同一定義）。バリア倍率/horizon も
  `data.labels` の定数を共有し、戦略を train/live で一致させる。
* 発注の非対称設計: エントリー/利確 = 指値(maker, post-only Alo)、損切り = 成行トリガ(taker)。
  退避は reduce-only。
* トレーディングループは GUI から独立。

確定仕様（user-confirmed）
-------------------------
* サイジング = ボラティリティターゲット（固定リスク）: risk$ = equity × risk_frac、
  size = risk$ / SL価格距離。σ に応じて建玉が自動調整。
* 発火 = 方向クラス確率 > 閾値。逆シグナルはドテン（決済→反対へ、各注文はゲート通過）。
* エグジット = TP指値 + SL成行トリガ + 時間決済（horizon 超で決済）。

純粋関数（サイジング/ブラケット/シグナル/アクション/SDK 写像/ゲート付き submit）は
単体テストできる。Hyperliquid SDK / asyncpg は遅延 import。
"""
from __future__ import annotations

import logging
import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

from data.labels import DEFAULT_HORIZON, DEFAULT_PT_MULT, DEFAULT_SL_MULT
from live.risk import Order, RiskConfig, RiskState, check_order
from shared import volatility

log = logging.getLogger("aevum.execution")


@dataclass(frozen=True)
class ExecConfig:
    risk_frac: float = 0.01            # 1トレードのリスク = equity の 1%
    prob_threshold: float = 0.5        # 方向クラス確率の発火閾値
    pt_mult: float = DEFAULT_PT_MULT   # labels と共有
    sl_mult: float = DEFAULT_SL_MULT
    horizon: int = DEFAULT_HORIZON
    entry_tif: str = "Alo"             # post-only（maker）
    risk: RiskConfig = field(default_factory=RiskConfig)


ExecutionPlan = namedtuple("ExecutionPlan", ["close", "entry", "brackets"])

# orders テーブルへの記録（order_id は IDENTITY 自動採番）。
ORDERS_INSERT_SQL = (
    "INSERT INTO orders "
    "(symbol, time, side, intent, order_type, price, size, status, cloid, risk_passed, reason) "
    "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
)


# --------------------------------------------------------------------------- #
# 純粋な意思決定
# --------------------------------------------------------------------------- #
def signal_from_probs(prob_down: float, prob_flat: float, prob_up: float, threshold: float) -> int:
    """方向シグナル {-1,0,1}。最大方向確率が閾値以上なら発火、なければ 0（様子見）。"""
    if prob_up >= threshold and prob_up >= prob_down:
        return 1
    if prob_down >= threshold and prob_down > prob_up:
        return -1
    return 0


def decide_action(signal: int, position_sign: int) -> str:
    """シグナルと現在建玉から行動を決める。"""
    if signal == 0:
        return "hold"
    if position_sign == 0:
        return "enter_long" if signal > 0 else "enter_short"
    if position_sign == signal:
        return "hold"  # 既に同方向 → 維持（ブラケット/時間で決済）
    return "reverse_long" if signal > 0 else "reverse_short"


def position_size(price: float, sigma: float, equity: float, *,
                  sl_mult: float, horizon: int, risk_frac: float) -> float:
    """固定リスクのサイズ（契約数）。size = (equity×risk_frac) / SL価格距離。"""
    if price <= 0 or sigma <= 0 or equity <= 0:
        return 0.0
    w = volatility.scale_to_horizon(sigma, horizon)     # σ × √horizon（labels と同式）
    sl_distance = price * (1.0 - math.exp(-sl_mult * w))  # entry→SL の価格距離
    if sl_distance <= 0:
        return 0.0
    return (equity * risk_frac) / sl_distance


def bracket_levels(entry_price: float, side: str, sigma: float, *,
                   pt_mult: float, sl_mult: float, horizon: int) -> tuple[float, float]:
    """(tp_level, sl_level)。対数空間で labels と同式。買い=TP上/SL下、売り=TP下/SL上。"""
    w = volatility.scale_to_horizon(sigma, horizon)
    if side == "buy":
        return entry_price * math.exp(pt_mult * w), entry_price * math.exp(-sl_mult * w)
    return entry_price * math.exp(-pt_mult * w), entry_price * math.exp(sl_mult * w)


def _close_order(symbol: str, position: float) -> Order:
    """建玉を成行で決済（reduce-only 相当）。stop_loss intent = 保護的成行退避。"""
    side = "sell" if position > 0 else "buy"
    return Order(symbol, side, "stop_loss", "market", abs(position), None)


def plan(prediction: dict, state: RiskState, config: ExecConfig = ExecConfig()) -> ExecutionPlan:
    """予測＋現在状態から実行プランを作る（close / entry / brackets）。発注はしない。"""
    sym = prediction["symbol"]
    mark = state.mark_prices.get(sym)
    sigma = float(prediction["sigma"])
    sig = signal_from_probs(prediction["prob_down"], prediction["prob_flat"],
                            prediction["prob_up"], config.prob_threshold)
    pos = state.positions.get(sym, 0.0)
    pos_sign = 0 if pos == 0 else (1 if pos > 0 else -1)
    action = decide_action(sig, pos_sign)

    close = entry = None
    brackets: list[Order] = []
    if action == "hold" or mark is None or mark <= 0:
        return ExecutionPlan(None, None, [])

    if action.startswith("reverse") and pos != 0:
        close = _close_order(sym, pos)

    if action in ("enter_long", "enter_short", "reverse_long", "reverse_short"):
        side = "buy" if action.endswith("long") else "sell"
        size = position_size(mark, sigma, state.equity,
                             sl_mult=config.sl_mult, horizon=config.horizon, risk_frac=config.risk_frac)
        if size > 0:
            entry = Order(sym, side, "entry", "limit", size, mark)
            tp, sl = bracket_levels(mark, side, sigma,
                                    pt_mult=config.pt_mult, sl_mult=config.sl_mult, horizon=config.horizon)
            exit_side = "sell" if side == "buy" else "buy"
            brackets = [
                Order(sym, exit_side, "take_profit", "limit", size, tp),
                Order(sym, exit_side, "stop_loss", "market", size, sl),  # price=sl は triggerPx
            ]
    return ExecutionPlan(close, entry, brackets)


# --------------------------------------------------------------------------- #
# SDK 写像
# --------------------------------------------------------------------------- #
def to_sdk_order(order: Order, *, tif: str = "Alo") -> dict:
    """risk.Order → hyperliquid exchange.order(**kwargs)。

    entry/take_profit = 指値(Alo, maker)、stop_loss = 成行トリガ(sl)。退避は reduce_only。
    """
    is_buy = order.side == "buy"
    reduce_only = order.intent != "entry"
    if order.intent == "stop_loss":
        if order.price is None:
            raise ValueError("stop_loss order needs a trigger price")
        order_type = {"trigger": {"triggerPx": float(order.price), "isMarket": True, "tpsl": "sl"}}
        limit_px = float(order.price)
    else:
        if order.price is None:
            raise ValueError(f"{order.intent} limit order needs a price")
        order_type = {"limit": {"tif": tif}}
        limit_px = float(order.price)
    return {
        "name": order.symbol,
        "is_buy": is_buy,
        "sz": float(order.size),
        "limit_px": limit_px,
        "order_type": order_type,
        "reduce_only": reduce_only,
    }


# --------------------------------------------------------------------------- #
# ゲート付き発注
# --------------------------------------------------------------------------- #
def _send(exchange, order: Order, tif: str):
    """承認済み注文を適切な SDK メソッドへ。

    * stop_loss かつ price なし = 即時成行クローズ → market_close
    * stop_loss かつ price あり = 待機ストップ(成行トリガ) → order(trigger)
    * entry / take_profit = 指値(Alo, maker) → order(limit)
    """
    if order.intent == "stop_loss" and order.price is None:
        return exchange.market_close(coin=order.symbol, sz=float(order.size))
    return exchange.order(**to_sdk_order(order, tif=tif))


def gate(order: Order, state: RiskState, config: ExecConfig):
    """注文をリスクゲートに通す。execution からの唯一の発注前関門。"""
    return check_order(order, state, config.risk)


async def submit(order: Order, state: RiskState, config: ExecConfig, exchange, conn=None) -> dict:
    """1注文をゲート→（承認時のみ）送信→記録。承認可否を必ず記録する。"""
    import datetime as _dt

    decision = gate(order, state, config)
    rec = {
        "symbol": order.symbol, "side": order.side, "intent": order.intent,
        "order_type": order.order_type, "price": order.price, "size": order.size,
        "risk_passed": decision.approved, "reason": decision.reason,
        "status": "pending" if decision.approved else "rejected",
        "exchange_response": None,
    }
    if decision.approved:
        rec["exchange_response"] = _send(exchange, order, config.entry_tif)
        rec["status"] = "open"
    else:
        log.warning("order rejected by risk gate: %s %s %s — %s",
                    order.symbol, order.side, order.intent, decision.reason)
    if conn is not None:
        await _record_order(conn, rec, _dt.datetime.now(_dt.timezone.utc))
    return rec


async def _record_order(conn, rec: dict, ts) -> None:
    await conn.execute(
        ORDERS_INSERT_SQL,
        rec["symbol"], ts, rec["side"], rec["intent"], rec["order_type"],
        rec["price"], rec["size"], rec["status"], None,
        rec["risk_passed"], rec["reason"],
    )


async def execute_plan(p: ExecutionPlan, state: RiskState, config: ExecConfig, exchange, conn=None) -> list[dict]:
    """close → entry の順に発注。brackets はエントリー約定後に place_brackets で発注する。"""
    recs = []
    if p.close is not None:
        recs.append(await submit(p.close, state, config, exchange, conn))
    if p.entry is not None:
        recs.append(await submit(p.entry, state, config, exchange, conn))
    return recs


async def place_brackets(p: ExecutionPlan, state: RiskState, config: ExecConfig, exchange, conn=None) -> list[dict]:
    """エントリー約定後に TP/SL ブラケットを発注（reduce-only）。"""
    recs = []
    for b in p.brackets:
        recs.append(await submit(b, state, config, exchange, conn))
    return recs
