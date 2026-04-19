from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import IndianDeliveryCostModel
from strategy.v9_engine import ALL, portfolio_returns


def test_indian_delivery_cost_model_matches_known_order_size_table() -> None:
    model = IndianDeliveryCostModel()

    buy_10k = model.order_cost(10_000.0, "BUY")
    sell_10k = model.order_cost(10_000.0, "SELL")
    buy_100k = model.order_cost(100_000.0, "BUY")
    sell_100k = model.order_cost(100_000.0, "SELL")

    assert buy_10k.bps == pytest_approx(23.67)
    assert sell_10k.bps == pytest_approx(45.77)
    assert buy_100k.bps == pytest_approx(14.23)
    assert sell_100k.bps == pytest_approx(15.09)
    assert (buy_100k.total + sell_100k.total) / 100_000.0 * 10_000.0 == pytest_approx(29.33)


def test_portfolio_returns_can_use_order_aware_delivery_costs() -> None:
    prices = pd.DataFrame(
        [
            {"NIFTY": 100.0, "MIDCAP": 100.0, "SMALLCAP": 100.0, "GOLD": 100.0, "SILVER": 100.0, "US": 100.0, "CASH": 1000.0},
            {"NIFTY": 101.0, "MIDCAP": 100.0, "SMALLCAP": 100.0, "GOLD": 100.0, "SILVER": 100.0, "US": 100.0, "CASH": 1000.0},
            {"NIFTY": 102.0, "MIDCAP": 100.0, "SMALLCAP": 100.0, "GOLD": 100.0, "SILVER": 100.0, "US": 100.0, "CASH": 1000.0},
        ],
        index=pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"]),
    )[ALL]
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    weights.loc["2026-04-01", "NIFTY"] = 1.0
    weights.loc["2026-04-02", "CASH"] = 1.0
    weights.loc["2026-04-03", "CASH"] = 1.0

    flat = portfolio_returns(prices, weights, tx_cost=0.003)
    official = portfolio_returns(
        prices,
        weights,
        tx_cost=0.003,
        cost_model=IndianDeliveryCostModel(),
        base_value=10_000.0,
    )

    assert official.iloc[0] < flat.iloc[0]


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=0.01)
