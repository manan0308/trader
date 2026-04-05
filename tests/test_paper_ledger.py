from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.paper_ledger import PaperLedgerPaths, recent_paper_history, update_paper_account_from_target


def test_same_day_rerun_keeps_pending_orders_unfilled_and_replaces_history_row(tmp_path: Path) -> None:
    paths = PaperLedgerPaths(
        state_path=tmp_path / "paper_state.json",
        history_path=tmp_path / "paper_history.jsonl",
        summary_path=tmp_path / "paper_summary.json",
    )
    prices = pd.DataFrame(
        [
            {
                "NIFTY": 300.0,
                "MIDCAP": 150.0,
                "SMALLCAP": 80.0,
                "GOLD": 70.0,
                "SILVER": 90.0,
                "US": 200.0,
                "CASH": 1_000.0,
            }
        ],
        index=[pd.Timestamp("2026-04-02")],
    )

    first = update_paper_account_from_target(
        {"NIFTY": 1.0},
        prices,
        initial_cash=10_000.0,
        paths=paths,
    )
    second = update_paper_account_from_target(
        {"NIFTY": 1.0},
        prices,
        initial_cash=10_000.0,
        paths=paths,
    )
    first_nifty = next(row for row in first["positions"] if row["asset"] == "NIFTY")
    second_nifty = next(row for row in second["positions"] if row["asset"] == "NIFTY")

    assert len(first["pending_orders"]) == 1
    assert first_nifty["quantity"] == 0
    assert second["recent_fills"] == []
    assert len(second["pending_orders"]) == 1
    assert second_nifty["quantity"] == 0

    history = recent_paper_history(limit=10, paths=paths)
    assert len(history) == 1
    assert history[0]["as_of"] == "2026-04-02"
