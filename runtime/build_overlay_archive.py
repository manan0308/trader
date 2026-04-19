#!/usr/bin/env python3
from __future__ import annotations

"""
Reconstruct a backtest-friendly overlay archive from saved live audit runs.

Why this exists:
- The live system re-decides the overlay every run.
- Naively unioning old multi-day overlay windows would overstate LLM impact,
  because an earlier override could keep leaking forward even after a later
  live run returned to neutral.

So this script extracts the *effective decision for each signal date* from the
saved audit snapshots and writes a one-day archive that can be replayed through
`load_llm_overlay(...)` for fair shadow-book backtests.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from runtime.audit_log import load_audit_runs
from runtime.store import write_json
from strategy.v9_engine import CACHE_DIR, RISKY


DEFAULT_OUTPUT = CACHE_DIR / "overlay_archive_from_audit_active_signal_days.json"
SNAPSHOT_KEYS = {
    "active": "overlay_snapshot",
    "policy": "policy_overlay_snapshot",
    "raw": "raw_overlay_snapshot",
}


def _zero_bias() -> Dict[str, float]:
    return {asset: 0.0 for asset in RISKY}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weights_changed(run: Dict[str, Any], tol: float = 1e-9) -> bool:
    base = run.get("v9_base_weights", {})
    final = run.get("v9_final_weights", {})
    for asset in set(RISKY + ["CASH"]):
        if abs(_float(base.get(asset, 0.0)) - _float(final.get(asset, 0.0))) > tol:
            return True
    return False


def _normalize_item(item: Dict[str, Any], signal_date: str, *, note: str) -> Dict[str, Any]:
    bias = _zero_bias()
    raw_bias = item.get("asset_bias", {})
    if isinstance(raw_bias, dict):
        for asset in RISKY:
            if asset in raw_bias:
                bias[asset] = max(-1.0, min(1.0, _float(raw_bias.get(asset, 0.0))))
    return {
        "date": signal_date,
        "holding_days": 1,
        "risk_off_override": max(0.0, min(1.0, _float(item.get("risk_off_override", 0.0)))),
        "asset_bias": bias,
        "rationale": str(item.get("rationale", note)).strip() or note,
        "confidence": max(0.0, min(1.0, _float(item.get("confidence", 0.0)))),
    }


def extract_effective_signal_day_item(snapshot: Dict[str, Any], signal_date: str) -> Dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None

    raw_dates = snapshot.get("dates", [])
    same_day_items = []
    if isinstance(raw_dates, list):
        for item in raw_dates:
            if isinstance(item, dict) and str(item.get("date", "")).strip() == signal_date:
                same_day_items.append(item)

    if same_day_items:
        return _normalize_item(
            same_day_items[-1],
            signal_date,
            note="reconstructed same-day overlay decision from audit snapshot",
        )

    default_risk = _float(snapshot.get("default_risk_off_override", 0.0))
    if default_risk > 0.0:
        return _normalize_item(
            {"risk_off_override": default_risk, "asset_bias": _zero_bias(), "confidence": 0.0},
            signal_date,
            note="reconstructed default overlay decision from audit snapshot",
        )

    return None


def build_overlay_archive(snapshot_key: str) -> Dict[str, Any]:
    runs = sorted(
        load_audit_runs(),
        key=lambda run: (
            str(run.get("signal_as_of") or run.get("as_of") or ""),
            str(run.get("ran_at") or ""),
        ),
    )

    items_by_date: Dict[str, Dict[str, Any]] = {}
    missing_changed_dates: List[str] = []
    contributing_runs = 0

    for run in runs:
        signal_date = str(run.get("signal_as_of") or run.get("as_of") or "").strip()
        if not signal_date:
            continue
        snapshot = run.get(snapshot_key, {})
        item = extract_effective_signal_day_item(snapshot, signal_date)
        if item is not None and _float(item.get("risk_off_override", 0.0)) > 0.0:
            items_by_date[signal_date] = item
            contributing_runs += 1
            continue
        if _weights_changed(run):
            missing_changed_dates.append(signal_date)

    dates = [items_by_date[key] for key in sorted(items_by_date)]
    return {
        "default_risk_off_override": 0.0,
        "dates": dates,
        "metadata": {
            "built_from": "audit_runs",
            "mode": "effective_signal_days",
            "snapshot_key": snapshot_key,
            "source_run_count": len(runs),
            "contributing_run_count": contributing_runs,
            "nonzero_signal_dates": len(dates),
            "missing_changed_dates": sorted(set(missing_changed_dates)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-day overlay archive from saved audit runs.")
    parser.add_argument(
        "--source",
        choices=sorted(SNAPSHOT_KEYS),
        default="active",
        help="Which audit snapshot to reconstruct: active (actual live), policy, or raw.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    snapshot_key = SNAPSHOT_KEYS[args.source]
    payload = build_overlay_archive(snapshot_key)
    output_path = Path(args.output).expanduser().resolve()
    write_json(output_path, payload)

    meta = payload.get("metadata", {})
    print(f"Saved overlay archive to {output_path}")
    print(
        "source_runs="
        f"{meta.get('source_run_count', 0)} "
        f"signal_dates={meta.get('nonzero_signal_dates', 0)} "
        f"missing_changed_dates={len(meta.get('missing_changed_dates', []))}"
    )


if __name__ == "__main__":
    main()
