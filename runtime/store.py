from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from strategy.v9_engine import CACHE_DIR


RUNTIME_DIR = CACHE_DIR
AUDIT_RUNS_PATH = CACHE_DIR / "audit_runs.jsonl"
PAPER_STATE_PATH = CACHE_DIR / "paper_trading_state.json"
PAPER_HISTORY_PATH = CACHE_DIR / "paper_trading_journal.jsonl"
PAPER_BASE_STATE_PATH = CACHE_DIR / "paper_base_state.json"
PAPER_BASE_HISTORY_PATH = CACHE_DIR / "paper_base_journal.jsonl"

PAPER_SUMMARY_PATH = CACHE_DIR / "paper_trading_latest.json"
PAPER_BASE_SUMMARY_PATH = CACHE_DIR / "paper_base_latest.json"
PAPER_COMPARISON_PATH = CACHE_DIR / "paper_comparison_latest.json"
LEARNING_STATE_PATH = CACHE_DIR / "learning_state.json"
RAW_OVERLAY_PATH = CACHE_DIR / "anthropic_overlay_latest.json"
EFFECTIVE_OVERLAY_PATH = CACHE_DIR / "active_overlay_latest.json"
DAILY_RUN_PATH = CACHE_DIR / "daily_cycle_latest.json"
EXECUTION_PLAN_BASE_PATH = CACHE_DIR / "execution_plan_base_latest.json"
EXECUTION_CONFIRMATION_PATH = CACHE_DIR / "execution_confirmation_latest.json"
EXECUTION_SUBMISSIONS_PATH = CACHE_DIR / "execution_submissions.jsonl"
EXECUTION_SUBMISSIONS_LATEST_PATH = CACHE_DIR / "execution_submissions_latest.json"
RECONCILIATION_PATH = CACHE_DIR / "reconciliation_latest.json"
RECONCILIATION_HISTORY_PATH = CACHE_DIR / "reconciliation_history.jsonl"


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


load_json = read_json


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[dict[str, Any]]:
    try:
        rows: List[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    except FileNotFoundError:
        return []


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
