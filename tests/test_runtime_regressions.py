from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.groww_adapter import actual_weight_snapshot
from execution.rebalance_core import current_portfolio_snapshot
from llm.anthropic_disagreement_review import build_payload
from runtime.paper_ledger import summarize_paper_state


def test_disagreement_review_payload_sets_tool_choice() -> None:
    payload = build_payload(
        model="claude-opus-4-6",
        prompt_text="prompt",
        packet={"foo": "bar"},
        enable_web_search=False,
        max_uses=1,
    )
    assert payload["tool_choice"] == {"type": "tool", "name": "submit_review"}


def test_cash_proxy_and_idle_cash_are_tracked_separately() -> None:
    prices = {
        "NIFTY": 100.0,
        "MIDCAP": 100.0,
        "SMALLCAP": 100.0,
        "GOLD": 100.0,
        "SILVER": 100.0,
        "US": 100.0,
        "CASH": 1_000.0,
    }
    holdings = {"CASH": 100}
    available_cash = 50_000.0

    snapshot = current_portfolio_snapshot(holdings, prices, available_cash)
    assert snapshot["weights"]["CASH"] == 100_000.0 / 150_000.0
    assert snapshot["idle_cash_weight"] == 50_000.0 / 150_000.0

    actual = actual_weight_snapshot(holdings, prices, available_cash)
    assert actual["weights"]["CASH"] == 100_000.0 / 150_000.0
    assert actual["idle_cash_weight"] == 50_000.0 / 150_000.0

    paper_state = {
        "as_of": "2026-04-02",
        "cash": available_cash,
        "positions": {"NIFTY": 0, "MIDCAP": 0, "SMALLCAP": 0, "GOLD": 0, "SILVER": 0, "US": 0, "CASH": 100},
        "pending_orders": [],
    }
    frame = pd.DataFrame([{asset: price for asset, price in prices.items()}], index=[pd.Timestamp("2026-04-02")])
    summary = summarize_paper_state(paper_state, {asset: float(frame.iloc[-1][asset]) for asset in frame.columns})
    assert summary["weights"]["CASH"] == 100_000.0 / 150_000.0
    assert summary["idle_cash_weight"] == 50_000.0 / 150_000.0
