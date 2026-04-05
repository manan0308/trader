from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.rebalance_core import ExecutionConfig, format_plan, plan_rebalance
from execution.groww_adapter import build_groww_order_requests, extract_asset_quantities, order_request_to_payload


def test_plan_rebalance_from_cash_builds_buy_orders() -> None:
    target = {
        "NIFTY": 0.40,
        "US": 0.30,
        "GOLD": 0.20,
        "CASH": 0.10,
    }
    prices = {
        "NIFTY": 300.0,
        "MIDCAP": 150.0,
        "SMALLCAP": 80.0,
        "GOLD": 70.0,
        "SILVER": 90.0,
        "US": 180.0,
        "CASH": 1_000.0,
    }
    plan = plan_rebalance(
        target_weights=target,
        holdings={},
        prices=prices,
        available_cash=100_000.0,
        config=ExecutionConfig(reserve_cash_value=0.0, reserve_cash_weight=0.0, min_order_value=1_000.0),
        as_of=pd.Timestamp("2026-04-03"),
    )
    assert plan.total_equity == 100_000.0
    assert {order.asset for order in plan.orders} >= {"NIFTY", "US", "GOLD", "CASH"}
    assert all(order.side == "BUY" for order in plan.orders)
    assert plan.post_trade_cash >= 0.0


def test_plan_rebalance_sells_before_buys() -> None:
    target = {
        "NIFTY": 0.20,
        "US": 0.50,
        "CASH": 0.30,
    }
    prices = {
        "NIFTY": 300.0,
        "MIDCAP": 150.0,
        "SMALLCAP": 80.0,
        "GOLD": 70.0,
        "SILVER": 90.0,
        "US": 200.0,
        "CASH": 1_000.0,
    }
    plan = plan_rebalance(
        target_weights=target,
        holdings={"NIFTY": 200, "US": 10},
        prices=prices,
        available_cash=2_000.0,
        config=ExecutionConfig(reserve_cash_value=0.0, reserve_cash_weight=0.0, min_order_value=1_000.0),
        as_of=pd.Timestamp("2026-04-03"),
    )
    assert plan.orders
    first_sell_index = next(i for i, order in enumerate(plan.orders) if order.side == "SELL")
    last_sell_index = max(i for i, order in enumerate(plan.orders) if order.side == "SELL")
    first_buy_index = next(i for i, order in enumerate(plan.orders) if order.side == "BUY")
    assert last_sell_index < first_buy_index or first_sell_index < first_buy_index


def test_extract_asset_quantities_from_groww_payloads() -> None:
    holdings_payload = {
        "holdings": [
            {"tradingSymbol": "NIFTYBEES", "quantity": 10},
            {"tradingSymbol": "MON100", "availableQuantity": 7},
        ]
    }
    positions_payload = {
        "positions": [
            {"tradingsymbol": "GOLDBEES", "netQty": 3},
        ]
    }
    quantities = extract_asset_quantities(holdings_payload, positions_payload)
    assert quantities["NIFTY"] == 10
    assert quantities["US"] == 7
    assert quantities["GOLD"] == 3


def test_build_groww_payloads_from_plan() -> None:
    target = {"NIFTY": 0.60, "CASH": 0.40}
    prices = {
        "NIFTY": 300.0,
        "MIDCAP": 150.0,
        "SMALLCAP": 80.0,
        "GOLD": 70.0,
        "SILVER": 90.0,
        "US": 200.0,
        "CASH": 1_000.0,
    }
    plan = plan_rebalance(
        target_weights=target,
        holdings={},
        prices=prices,
        available_cash=50_000.0,
        config=ExecutionConfig(reserve_cash_value=0.0, reserve_cash_weight=0.0, min_order_value=1_000.0),
    )
    requests = build_groww_order_requests(plan.orders)
    payloads = [order_request_to_payload(request) for request in requests]
    assert payloads
    assert all("trading_symbol" in payload for payload in payloads)
    assert all(payload["transaction_type"] in {"BUY", "SELL"} for payload in payloads)


def test_plan_rebalance_rejects_non_positive_price() -> None:
    target = {"NIFTY": 0.5, "CASH": 0.5}
    prices = {
        "NIFTY": 0.0,  # stale / missing quote
        "MIDCAP": 150.0,
        "SMALLCAP": 80.0,
        "GOLD": 70.0,
        "SILVER": 90.0,
        "US": 200.0,
        "CASH": 1_000.0,
    }
    with pytest.raises(ValueError, match="Non-positive prices"):
        plan_rebalance(
            target_weights=target,
            holdings={},
            prices=prices,
            available_cash=10_000.0,
            config=ExecutionConfig(reserve_cash_value=0.0, reserve_cash_weight=0.0, min_order_value=1_000.0),
        )


def test_format_plan_has_order_summary() -> None:
    target = {"NIFTY": 1.0}
    prices = {
        "NIFTY": 300.0,
        "MIDCAP": 150.0,
        "SMALLCAP": 80.0,
        "GOLD": 70.0,
        "SILVER": 90.0,
        "US": 200.0,
        "CASH": 1_000.0,
    }
    plan = plan_rebalance(
        target_weights=target,
        holdings={},
        prices=prices,
        available_cash=10_000.0,
        config=ExecutionConfig(reserve_cash_value=0.0, reserve_cash_weight=0.0, min_order_value=1_000.0),
    )
    text = format_plan(plan)
    assert "Orders:" in text
    assert "BUY" in text
