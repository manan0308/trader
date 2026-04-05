from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.audit_log import merge_audit_records
from runtime.store import _atomic_write_text, read_json, write_json


def test_atomic_write_replaces_file_and_cleans_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    write_json(target, {"a": 1})
    assert read_json(target) == {"a": 1}

    # Overwriting should replace atomically without leaving stray temp files.
    write_json(target, {"a": 2})
    assert read_json(target) == {"a": 2}

    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".payload.json.")]
    assert leftovers == []


def test_atomic_write_preserves_prior_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"
    write_json(target, {"version": 1})
    original = target.read_text(encoding="utf-8")

    class Boom(RuntimeError):
        pass

    # Simulate a crash mid-write by aborting before ``os.replace`` runs: the
    # prior file must remain intact and no sibling temp file should be left
    # behind.
    import runtime.store as store_mod

    real_replace = store_mod.os.replace

    def failing_replace(src: str, dst: str) -> None:
        raise Boom("simulated crash before rename")

    store_mod.os.replace = failing_replace  # type: ignore[assignment]
    try:
        try:
            _atomic_write_text(target, "corrupted-bytes")
        except Boom:
            pass
    finally:
        store_mod.os.replace = real_replace  # type: ignore[assignment]

    # Original payload survived and no ``.payload.json.*.tmp`` leaked.
    assert target.read_text(encoding="utf-8") == original
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".payload.json.")]
    assert leftovers == []


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
