"""
Minimal strategy template for strategy_tester.py.

You can replace the logic below with your own rules and keep the same
function names so the tester can auto-detect them.
"""

from __future__ import annotations

import pandas as pd

RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]


def fetch_data(start: str = "2012-01-01") -> pd.DataFrame:
    # Optional. If you omit this, strategy_tester.py will use the shared
    # cleaned INR data source from trader_system.strategy.v9_engine.
    raise NotImplementedError


def run_strategy(prices: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Return a daily weights DataFrame with the same index as `prices`.
    This template uses a simple equal-weight risky basket so the tester
    produces a sensible baseline before you replace the logic.
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    cash = float((params or {}).get("cash", 0.0))
    risky = max(0.0, 1.0 - cash)
    for asset in RISKY:
        weights[asset] = risky / len(RISKY)
    weights["CASH"] = cash
    return weights


def backtest(prices: pd.DataFrame, params: dict | None = None):
    # Optional. If present, the tester can use this directly.
    return run_strategy(prices, params=params)
