from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from trader_system.strategy.v9_engine import CACHE_DIR


AUDIT_LATEST_PATH = CACHE_DIR / "audit_runs_latest.json"
AUDIT_JSONL_PATH = CACHE_DIR / "audit_runs.jsonl"


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def load_audit_runs() -> List[Dict[str, Any]]:
    try:
        if AUDIT_LATEST_PATH.exists():
            return json.loads(AUDIT_LATEST_PATH.read_text(encoding="utf-8"))
        if AUDIT_JSONL_PATH.exists():
            rows = []
            for line in AUDIT_JSONL_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
    except Exception:
        pass
    return []


def write_audit_runs(runs: List[Dict[str, Any]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = runs[-250:]
    AUDIT_LATEST_PATH.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    with AUDIT_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in trimmed:
            handle.write(stable_json(row))
            handle.write("\n")


def merge_audit_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    merged.update(incoming)

    for key in ("evaluations", "submission", "reconciliation"):
        if key in existing and key not in incoming:
            merged[key] = existing[key]

    if "execution_summary" in existing and "execution_summary" in incoming:
        execution_summary = dict(existing.get("execution_summary", {}))
        execution_summary.update(incoming.get("execution_summary", {}))
        merged["execution_summary"] = execution_summary

    return merged


def upsert_audit_run(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = load_audit_runs()
    run_key = str(run.get("run_key", ""))
    replaced = False
    for idx, existing in enumerate(runs):
        if str(existing.get("run_key", "")) == run_key and run_key:
            runs[idx] = merge_audit_records(existing, run)
            replaced = True
            break
    if not replaced:
        runs.append(run)
    write_audit_runs(runs)
    return runs


def update_audit_run(run_key: str, patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not run_key:
        return load_audit_runs()
    runs = load_audit_runs()
    updated = False
    for idx, existing in enumerate(runs):
        if str(existing.get("run_key", "")) == str(run_key):
            runs[idx] = merge_audit_records(existing, patch)
            updated = True
            break
    if updated:
        write_audit_runs(runs)
    return runs
