from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.audit_log import merge_audit_records


def test_merge_audit_records_preserves_evaluations_and_updates_submission() -> None:
    existing = {
        "run_key": "abc123",
        "status": "ok",
        "evaluations": {"1d": {"final_return": 0.01}},
        "execution_summary": {"orders": 3, "portfolio_value": 100000},
        "submission": {"status": "preview"},
    }
    incoming = {
        "run_key": "abc123",
        "status": "ok",
        "execution_summary": {"orders": 5},
    }

    merged = merge_audit_records(existing, incoming)

    assert merged["evaluations"] == existing["evaluations"]
    assert merged["submission"] == existing["submission"]
    assert merged["execution_summary"]["orders"] == 5
    assert merged["execution_summary"]["portfolio_value"] == 100000
