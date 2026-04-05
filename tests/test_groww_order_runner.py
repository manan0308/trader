from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader_system.execution.groww_order_runner import build_reference_id, merged_plan_orders


def test_merged_plan_orders_keeps_order_and_payloads() -> None:
    payload = {
        "as_of": "2026-04-02",
        "plan": {
            "orders": [
                {"asset": "NIFTY", "side": "BUY", "quantity": 10},
                {"asset": "US", "side": "SELL", "quantity": 2},
            ],
            "groww_payloads": [
                {"trading_symbol": "NIFTYBEES", "transaction_type": "BUY"},
                {"trading_symbol": "MON100", "transaction_type": "SELL"},
            ],
        },
    }

    rows = merged_plan_orders(payload)

    assert len(rows) == 2
    assert rows[0]["sequence"] == 1
    assert rows[0]["groww_payload"]["trading_symbol"] == "NIFTYBEES"
    assert rows[1]["groww_payload"]["trading_symbol"] == "MON100"


def test_build_reference_id_is_deterministic() -> None:
    assert build_reference_id("2026-04-02", "abcdef1234567890", 3) == "TRDR-20260402-abcdef12-03"
