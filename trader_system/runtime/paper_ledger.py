from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

from trader_system.strategy.v9_engine import ALL
from trader_system.execution.rebalance_core import ExecutionConfig, plan_rebalance
from trader_system.runtime.display_labels import asset_label
from trader_system.runtime.store import (
    PAPER_BASE_HISTORY_PATH,
    PAPER_BASE_STATE_PATH,
    PAPER_BASE_SUMMARY_PATH,
    PAPER_HISTORY_PATH,
    PAPER_STATE_PATH,
    PAPER_SUMMARY_PATH,
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
)


DEFAULT_PAPER_STARTING_CASH = 1_000_000.0
DEFAULT_ESTIMATED_COST_BPS = 30.0


@dataclass(frozen=True)
class PaperLedgerPaths:
    state_path: Path
    history_path: Path
    summary_path: Path


DEFAULT_PAPER_PATHS = PaperLedgerPaths(
    state_path=PAPER_STATE_PATH,
    history_path=PAPER_HISTORY_PATH,
    summary_path=PAPER_SUMMARY_PATH,
)
BASELINE_PAPER_PATHS = PaperLedgerPaths(
    state_path=PAPER_BASE_STATE_PATH,
    history_path=PAPER_BASE_HISTORY_PATH,
    summary_path=PAPER_BASE_SUMMARY_PATH,
)

PAPER_LATEST_PATH = DEFAULT_PAPER_PATHS.summary_path
PAPER_JOURNAL_PATH = DEFAULT_PAPER_PATHS.history_path


def default_paper_state(starting_cash: float = DEFAULT_PAPER_STARTING_CASH) -> Dict[str, Any]:
    return {
        "version": 1,
        "as_of": None,
        "cash": float(starting_cash),
        "positions": {asset: 0 for asset in ALL},
        "pending_orders": [],
    }


def load_paper_state(
    starting_cash: float = DEFAULT_PAPER_STARTING_CASH,
    *,
    paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS,
) -> Dict[str, Any]:
    state = read_json(paths.state_path, default=None)
    if not isinstance(state, dict):
        state = default_paper_state(starting_cash=starting_cash)
    state.setdefault("version", 1)
    state.setdefault("as_of", None)
    state.setdefault("cash", float(starting_cash))
    state.setdefault("positions", {asset: 0 for asset in ALL})
    state.setdefault("pending_orders", [])
    for asset in ALL:
        state["positions"].setdefault(asset, 0)
    return state


def save_paper_state(state: Dict[str, Any], *, paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS) -> None:
    write_json(paths.state_path, state)


def price_map_from_prices(prices: pd.DataFrame) -> Dict[str, float]:
    latest = prices.iloc[-1]
    return {asset: float(latest[asset]) for asset in ALL}


def fill_pending_orders(
    state: Dict[str, Any],
    price_map: Mapping[str, float],
    as_of: str,
    estimated_cost_bps: float = DEFAULT_ESTIMATED_COST_BPS,
) -> List[Dict[str, Any]]:
    positions = {asset: int(state["positions"].get(asset, 0)) for asset in ALL}
    cash = float(state.get("cash", 0.0))
    pending = list(state.get("pending_orders", []))
    filled: List[Dict[str, Any]] = []

    for order in pending:
        asset = str(order.get("asset", "")).upper()
        if asset not in ALL:
            continue
        quantity = max(int(order.get("quantity", 0)), 0)
        if quantity <= 0:
            continue
        side = str(order.get("side", "")).upper()
        fill_price = float(price_map.get(asset, order.get("reference_price", 0.0)))
        gross = quantity * fill_price
        fees = gross * estimated_cost_bps / 10_000.0

        if side == "BUY":
            affordable = gross + fees <= cash + 1e-9
            if not affordable:
                continue
            positions[asset] += quantity
            cash -= gross + fees
        elif side == "SELL":
            sell_qty = min(quantity, positions[asset])
            if sell_qty <= 0:
                continue
            gross = sell_qty * fill_price
            fees = gross * estimated_cost_bps / 10_000.0
            positions[asset] -= sell_qty
            cash += gross - fees
            quantity = sell_qty
        else:
            continue

        filled.append(
            {
                "filled_at": as_of,
                "asset": asset,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "gross": gross,
                "fees": fees,
            }
        )

    state["positions"] = positions
    state["cash"] = float(cash)
    state["pending_orders"] = []
    state["as_of"] = as_of
    return filled


def queue_pending_orders(state: Dict[str, Any], orders: Iterable[Mapping[str, Any]], created_at: str) -> None:
    queued: List[Dict[str, Any]] = []
    for order in orders:
        quantity = max(int(order.get("quantity", 0)), 0)
        if quantity <= 0:
            continue
        queued.append(
            {
                "created_at": created_at,
                "asset": str(order.get("asset", "")).upper(),
                "side": str(order.get("side", "")).upper(),
                "quantity": quantity,
                "reference_price": float(order.get("reference_price", 0.0)),
                "target_weight": float(order.get("delta_weight", 0.0)),
            }
        )
    state["pending_orders"] = queued


def summarize_paper_state(
    state: Mapping[str, Any],
    price_map: Mapping[str, float],
    recent_fills: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    cash = float(state.get("cash", 0.0))
    positions = {asset: int(state.get("positions", {}).get(asset, 0)) for asset in ALL}
    market_values = {asset: positions[asset] * float(price_map.get(asset, 0.0)) for asset in ALL}
    holdings_value = float(sum(market_values.values()))
    equity = cash + holdings_value

    weights = {
        asset: (market_values[asset] / equity if equity > 0 else 0.0)
        for asset in ALL
    }
    weights["CASH"] = cash / equity if equity > 0 else 1.0

    def labeled_order(row: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(row)
        asset = str(payload.get("asset", "")).upper()
        if asset:
            payload["asset"] = asset_label(asset)
        return payload

    return {
        "as_of": state.get("as_of"),
        "cash": cash,
        "equity": equity,
        "total_equity": equity,
        "holdings_value": holdings_value,
        "weights": weights,
        "positions": [
            {
                "asset": asset_label(asset),
                "quantity": positions[asset],
                "price": float(price_map.get(asset, 0.0)),
                "market_value": market_values[asset],
            }
            for asset in ALL
        ],
        "position_count": sum(1 for asset in ALL if positions[asset] > 0),
        "pending_orders": [labeled_order(row) for row in list(state.get("pending_orders", []))],
        "recent_fills": [labeled_order(row) for row in list(recent_fills or [])],
    }


def append_paper_history(
    summary: Mapping[str, Any],
    filled_orders: Iterable[Mapping[str, Any]],
    queued_orders: Iterable[Mapping[str, Any]],
    *,
    paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS,
) -> Dict[str, Any]:
    row = {
        "as_of": summary.get("as_of"),
        "equity": float(summary.get("equity", 0.0)),
        "cash": float(summary.get("cash", 0.0)),
        "weights": summary.get("weights", {}),
        "filled_count": len(list(filled_orders)),
        "queued_count": len(list(queued_orders)),
    }
    append_jsonl(paths.history_path, row)
    return row


def recent_paper_history(limit: int = 30, *, paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS) -> List[Dict[str, Any]]:
    rows = read_jsonl(paths.history_path)
    return rows[-limit:]


def write_paper_summary(summary: Dict[str, Any], history_limit: int = 30, *, paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS) -> None:
    payload = {
        **summary,
        "history": recent_paper_history(limit=history_limit, paths=paths),
    }
    write_json(paths.summary_path, payload)


def _update_paper_from_orders(
    planned_orders: Iterable[Mapping[str, Any]],
    prices: pd.DataFrame,
    initial_cash: float,
    *,
    paths: PaperLedgerPaths,
) -> Dict[str, Any]:
    state = load_paper_state(starting_cash=initial_cash, paths=paths)
    price_map = price_map_from_prices(prices)
    as_of = prices.index[-1].strftime("%Y-%m-%d")

    fills = fill_pending_orders(state, price_map=price_map, as_of=as_of)
    planned_orders = list(planned_orders)
    queue_pending_orders(state, planned_orders, created_at=as_of)
    summary = summarize_paper_state(state, price_map=price_map, recent_fills=fills)
    history_row = append_paper_history(summary, fills, planned_orders, paths=paths)
    save_paper_state(state, paths=paths)

    history = recent_paper_history(limit=252, paths=paths)
    first_equity = float(history[0]["equity"]) if history else float(summary["equity"])
    latest_equity = float(summary["equity"])
    net_pnl = latest_equity - first_equity
    return_since_start = latest_equity / first_equity - 1.0 if first_equity > 0 else 0.0

    latest_payload = {
        **summary,
        "history": history[-30:],
        "equity_curve": history[-30:],
        "latest_history_row": history_row,
        "total_equity": latest_equity,
        "net_pnl": net_pnl,
        "return_since_start": return_since_start,
    }
    write_json(paths.summary_path, latest_payload)
    return latest_payload


def update_paper_account(
    plan_payload: Mapping[str, Any],
    prices: pd.DataFrame,
    initial_cash: float = DEFAULT_PAPER_STARTING_CASH,
    *,
    paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS,
) -> Dict[str, Any]:
    planned_orders = plan_payload.get("plan", {}).get("orders", []) if isinstance(plan_payload, Mapping) else []
    return _update_paper_from_orders(planned_orders, prices, initial_cash, paths=paths)


def target_orders_for_state(
    target_weights: Mapping[str, float],
    prices: pd.DataFrame,
    *,
    starting_cash: float = DEFAULT_PAPER_STARTING_CASH,
    paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS,
) -> List[Dict[str, Any]]:
    state = load_paper_state(starting_cash=starting_cash, paths=paths)
    price_map = price_map_from_prices(prices)
    working_state = {
        "positions": dict(state.get("positions", {})),
        "cash": float(state.get("cash", starting_cash)),
        "pending_orders": list(state.get("pending_orders", [])),
        "as_of": state.get("as_of"),
    }
    fill_pending_orders(working_state, price_map=price_map, as_of=prices.index[-1].strftime("%Y-%m-%d"))
    plan = plan_rebalance(
        target_weights={asset: float(target_weights.get(asset, 0.0)) for asset in ALL},
        holdings=working_state["positions"],
        prices=price_map,
        available_cash=float(working_state["cash"]),
        config=ExecutionConfig(),
        as_of=prices.index[-1],
    )
    return [
        {
            "asset": order.asset,
            "side": order.side,
            "quantity": order.quantity,
            "reference_price": order.reference_price,
            "order_price": order.order_price,
            "current_quantity": order.current_quantity,
            "target_quantity": order.target_quantity,
            "current_value": order.current_value,
            "target_value": order.target_value,
            "delta_value": order.delta_value,
            "estimated_cost": order.estimated_cost,
            "notes": order.notes,
            "delta_weight": float(plan.target_weights[order.asset] - plan.starting_weights[order.asset]),
        }
        for order in plan.orders
    ]


def update_paper_account_from_target(
    target_weights: Mapping[str, float],
    prices: pd.DataFrame,
    initial_cash: float = DEFAULT_PAPER_STARTING_CASH,
    *,
    paths: PaperLedgerPaths = DEFAULT_PAPER_PATHS,
) -> Dict[str, Any]:
    planned_orders = target_orders_for_state(target_weights, prices, starting_cash=initial_cash, paths=paths)
    return _update_paper_from_orders(planned_orders, prices, initial_cash, paths=paths)
