#!/usr/bin/env python3
"""
Groww order runner
==================

Safe broker-facing wrapper around the latest execution plan.

Modes:
- dry-run: inspect and persist the plan that would be sent
- confirm: explicitly approve a specific plan hash for placement
- place: submit orders using deterministic reference ids
- reconcile: compare target quantities/weights against live broker holdings
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from trader_system.runtime.audit_log import stable_hash, update_audit_run
from trader_system.execution.order_planner import PLAN_PATH
from trader_system.execution.groww_adapter import (
    actual_weight_snapshot,
    current_holdings_snapshot,
    groww_price_map,
)
from trader_system.broker.groww_client import PRODUCTION_GROWW_UNIVERSE_PATH, GrowwSession, load_production_groww_universe
from trader_system.runtime.india_market_calendar import market_clock
from trader_system.runtime.store import (
    DAILY_RUN_PATH,
    EXECUTION_CONFIRMATION_PATH,
    EXECUTION_SUBMISSIONS_LATEST_PATH,
    EXECUTION_SUBMISSIONS_PATH,
    RECONCILIATION_HISTORY_PATH,
    RECONCILIATION_PATH,
    append_jsonl,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def load_plan(path: Path) -> Dict[str, Any]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict) or not payload.get("plan"):
        raise RuntimeError(f"Execution plan missing or invalid at {path}")
    return payload


def plan_key(payload: Mapping[str, Any]) -> str:
    existing = str(payload.get("plan_key", "")).strip()
    if existing:
        return existing
    return stable_hash(
        {
            "as_of": payload.get("as_of"),
            "broker_map": payload.get("broker_map", {}),
            "target_weights": payload.get("plan", {}).get("target_weights", {}),
            "orders": payload.get("plan", {}).get("groww_payloads", []),
        }
    )[:16]


def load_daily_run_key(plan_payload: Mapping[str, Any]) -> str:
    daily = read_json(DAILY_RUN_PATH, default={})
    if isinstance(daily, dict):
        key = str(daily.get("run_key", "")).strip()
        if key and str(daily.get("as_of", "")) == str(plan_payload.get("as_of", "")):
            return key
    return str(plan_payload.get("decision_run_key", "")).strip()


def merged_plan_orders(plan_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    plan_block = plan_payload.get("plan", {}) if isinstance(plan_payload, Mapping) else {}
    order_rows = plan_block.get("orders", []) if isinstance(plan_block, Mapping) else []
    groww_rows = plan_block.get("groww_payloads", []) if isinstance(plan_block, Mapping) else []
    if not isinstance(order_rows, list) or not isinstance(groww_rows, list):
        return []

    merged: List[Dict[str, Any]] = []
    for idx, order in enumerate(order_rows):
        if not isinstance(order, dict):
            continue
        payload = groww_rows[idx] if idx < len(groww_rows) and isinstance(groww_rows[idx], dict) else {}
        row = dict(order)
        row["groww_payload"] = payload
        row["sequence"] = idx + 1
        merged.append(row)
    return merged


def build_reference_id(as_of: str, key: str, sequence: int) -> str:
    compact = as_of.replace("-", "")
    return f"TRDR-{compact}-{key[:8]}-{sequence:02d}"


def load_confirmation() -> Dict[str, Any]:
    payload = read_json(EXECUTION_CONFIRMATION_PATH, default={})
    return payload if isinstance(payload, dict) else {}


def write_confirmation(payload: Dict[str, Any]) -> None:
    write_json(EXECUTION_CONFIRMATION_PATH, payload)


def load_submission_history() -> List[Dict[str, Any]]:
    return read_jsonl(EXECUTION_SUBMISSIONS_PATH)


def write_submission_history(rows: List[Dict[str, Any]]) -> None:
    trimmed = rows[-250:]
    write_jsonl(EXECUTION_SUBMISSIONS_PATH, trimmed)
    write_json(EXECUTION_SUBMISSIONS_LATEST_PATH, trimmed)


def upsert_submission(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = load_submission_history()
    submission_key = str(row.get("submission_key", "")).strip()
    replaced = False
    for idx, existing in enumerate(rows):
        if str(existing.get("submission_key", "")).strip() == submission_key and submission_key:
            rows[idx] = {**existing, **row}
            replaced = True
            break
    if not replaced:
        rows.append(row)
    write_submission_history(rows)
    return rows


def append_reconciliation(row: Dict[str, Any]) -> None:
    append_jsonl(RECONCILIATION_HISTORY_PATH, row)
    history = read_jsonl(RECONCILIATION_HISTORY_PATH)[-250:]
    write_jsonl(RECONCILIATION_HISTORY_PATH, history)
    write_json(RECONCILIATION_PATH, row)


def pretty_amount(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def ensure_place_allowed(*, allow_closed_market: bool) -> None:
    clock = market_clock()
    if allow_closed_market:
        return
    if not clock.is_trading_day:
        raise RuntimeError(f"Market is closed today: {clock.holiday_name or clock.session}")
    if clock.session not in {"pre_open", "regular", "closing"}:
        raise RuntimeError(f"Market is not in a tradable session right now: {clock.session}")


def dry_run_payload(plan_payload: Mapping[str, Any], audit_run_key: str) -> Dict[str, Any]:
    orders = merged_plan_orders(plan_payload)
    total_buy = sum(float(row.get("delta_value", 0.0)) for row in orders if str(row.get("side", "")).upper() == "BUY")
    total_sell = sum(float(row.get("delta_value", 0.0)) for row in orders if str(row.get("side", "")).upper() == "SELL")
    row = {
        "submission_key": stable_hash({"mode": "dry-run", "plan_key": plan_key(plan_payload)})[:16],
        "mode": "dry-run",
        "status": "preview",
        "created_at": now_iso(),
        "as_of": plan_payload.get("as_of"),
        "plan_key": plan_key(plan_payload),
        "decision_run_key": audit_run_key,
        "order_count": len(orders),
        "gross_buy_value": total_buy,
        "gross_sell_value": total_sell,
        "orders": orders,
    }
    upsert_submission(row)
    if audit_run_key:
        update_audit_run(audit_run_key, {"submission": row})
    return row


def confirm_payload(plan_payload: Mapping[str, Any], audit_run_key: str, note: str) -> Dict[str, Any]:
    row = {
        "confirmed_at": now_iso(),
        "as_of": plan_payload.get("as_of"),
        "plan_key": plan_key(plan_payload),
        "decision_run_key": audit_run_key,
        "note": note,
        "order_count": len(merged_plan_orders(plan_payload)),
    }
    write_confirmation(row)
    if audit_run_key:
        update_audit_run(audit_run_key, {"submission": {"confirmation": row}})
    return row


def fetch_existing_reference_status(session: GrowwSession, reference_id: str, segment: str) -> Optional[Dict[str, Any]]:
    try:
        response = session.client.get_order_status_by_reference(segment=segment, order_reference_id=reference_id)
    except Exception:
        return None
    return response if isinstance(response, dict) and response else None


def place_payload(
    plan_payload: Mapping[str, Any],
    session: GrowwSession,
    *,
    allow_closed_market: bool,
    require_confirm: bool,
    confirmation_note: str,
) -> Dict[str, Any]:
    ensure_place_allowed(allow_closed_market=allow_closed_market)

    pk = plan_key(plan_payload)
    audit_run_key = load_daily_run_key(plan_payload)
    confirmation = load_confirmation()
    if require_confirm and str(confirmation.get("plan_key", "")).strip() != pk:
        raise RuntimeError(
            "Plan is not confirmed. Run `python -m trader_system.execution.groww_order_runner --mode confirm` first."
        )

    orders = merged_plan_orders(plan_payload)
    results: List[Dict[str, Any]] = []
    placed_count = 0
    skipped_count = 0
    failed_count = 0

    for order in orders:
        groww_payload = order.get("groww_payload", {}) if isinstance(order.get("groww_payload"), dict) else {}
        sequence = int(order.get("sequence", 0))
        reference_id = build_reference_id(str(plan_payload.get("as_of", "")), pk, sequence)
        existing = fetch_existing_reference_status(session, reference_id, str(groww_payload.get("segment", "CASH")))
        if existing:
            skipped_count += 1
            results.append(
                {
                    "asset": order.get("asset"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "order_reference_id": reference_id,
                    "status": "skipped_existing",
                    "existing_status": existing,
                }
            )
            continue

        try:
            response = session.client.place_order(
                validity=str(groww_payload.get("validity", "DAY")),
                exchange=str(groww_payload.get("exchange", "NSE")),
                order_type=str(groww_payload.get("order_type", "LIMIT")),
                product=str(groww_payload.get("product", "CNC")),
                quantity=int(groww_payload.get("quantity", 0)),
                segment=str(groww_payload.get("segment", "CASH")),
                trading_symbol=str(groww_payload.get("trading_symbol")),
                transaction_type=str(groww_payload.get("transaction_type")),
                order_reference_id=reference_id,
                price=float(groww_payload.get("price", 0.0) or 0.0),
            )
            placed_count += 1
            results.append(
                {
                    "asset": order.get("asset"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "order_reference_id": reference_id,
                    "status": "submitted",
                    "response": response,
                }
            )
        except Exception as exc:
            failed_count += 1
            results.append(
                {
                    "asset": order.get("asset"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "order_reference_id": reference_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    submission = {
        "submission_key": stable_hash({"mode": "place", "plan_key": pk, "results": results})[:16],
        "mode": "place",
        "status": "ok" if failed_count == 0 else ("partial" if placed_count > 0 or skipped_count > 0 else "failed"),
        "created_at": now_iso(),
        "as_of": plan_payload.get("as_of"),
        "plan_key": pk,
        "decision_run_key": audit_run_key,
        "confirmation": confirmation if require_confirm else {"note": confirmation_note, "skipped": True},
        "order_count": len(orders),
        "placed_count": placed_count,
        "skipped_existing_count": skipped_count,
        "failed_count": failed_count,
        "orders": results,
    }
    upsert_submission(submission)
    if audit_run_key:
        update_audit_run(audit_run_key, {"submission": submission})
    return submission


def reconcile_payload(plan_payload: Mapping[str, Any], session: GrowwSession) -> Dict[str, Any]:
    instruments = load_production_groww_universe(plan_payload.get("groww_universe_file"))
    snapshot = current_holdings_snapshot(session, instrument_map=instruments)
    fallback_prices = {
        asset: float(order.get("reference_price", 0.0))
        for order in merged_plan_orders(plan_payload)
        if order.get("asset")
    }
    live_prices = groww_price_map(
        session,
        instrument_map=instruments,
        fallback_prices=fallback_prices,
    )
    actual = actual_weight_snapshot(
        quantities=snapshot["quantities"],
        price_map=live_prices,
        available_cash=float(snapshot.get("available_cash") or 0.0),
    )
    target_quantities = plan_payload.get("plan", {}).get("target_quantities", {})
    target_weights = plan_payload.get("plan", {}).get("target_weights", {})

    drifts = []
    for asset, target_qty in sorted(target_quantities.items()):
        actual_qty = int(actual["quantities"].get(asset, 0))
        target_weight = float(target_weights.get(asset, 0.0))
        actual_weight = float(actual["weights"].get(asset, 0.0))
        drifts.append(
            {
                "asset": asset,
                "target_quantity": int(target_qty),
                "actual_quantity": actual_qty,
                "quantity_gap": actual_qty - int(target_qty),
                "target_weight": target_weight,
                "actual_weight": actual_weight,
                "weight_gap": actual_weight - target_weight,
            }
        )

    latest_submission = {}
    for row in reversed(load_submission_history()):
        if str(row.get("plan_key", "")) == plan_key(plan_payload):
            latest_submission = row
            break

    reconciliation = {
        "reconciled_at": now_iso(),
        "as_of": plan_payload.get("as_of"),
        "plan_key": plan_key(plan_payload),
        "decision_run_key": load_daily_run_key(plan_payload),
        "broker_cash_available": float(snapshot.get("available_cash") or 0.0),
        "broker_snapshot": snapshot,
        "actual_snapshot": actual,
        "drift_rows": drifts,
        "latest_submission": latest_submission,
    }
    append_reconciliation(reconciliation)
    audit_run_key = str(reconciliation.get("decision_run_key", "")).strip()
    if audit_run_key:
        update_audit_run(audit_run_key, {"reconciliation": reconciliation})
    return reconciliation


def print_summary(title: str, payload: Mapping[str, Any]) -> None:
    print(title)
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe Groww order runner for the latest execution plan.")
    parser.add_argument("--mode", choices=["dry-run", "confirm", "place", "reconcile"], required=True)
    parser.add_argument("--plan-file", default=str(PLAN_PATH))
    parser.add_argument("--groww-universe-file")
    parser.add_argument("--note", default="")
    parser.add_argument("--skip-confirm", action="store_true")
    parser.add_argument("--allow-closed-market", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan_file).expanduser().resolve()
    plan_payload = load_plan(plan_path)
    if args.groww_universe_file:
        plan_payload["groww_universe_file"] = args.groww_universe_file
    elif not plan_payload.get("groww_universe_file"):
        plan_payload["groww_universe_file"] = str(PRODUCTION_GROWW_UNIVERSE_PATH)

    audit_run_key = load_daily_run_key(plan_payload)

    if args.mode == "dry-run":
        payload = dry_run_payload(plan_payload, audit_run_key)
        return print_summary("Dry-run preview", payload)

    if args.mode == "confirm":
        payload = confirm_payload(plan_payload, audit_run_key, note=args.note)
        return print_summary("Confirmation saved", payload)

    session = GrowwSession.from_env(
        instrument_map=load_production_groww_universe(plan_payload.get("groww_universe_file")),
    )

    if args.mode == "place":
        payload = place_payload(
            plan_payload,
            session,
            allow_closed_market=args.allow_closed_market,
            require_confirm=not args.skip_confirm,
            confirmation_note=args.note,
        )
        return print_summary("Placement result", payload)

    payload = reconcile_payload(plan_payload, session)
    print_summary("Reconciliation result", payload)


if __name__ == "__main__":
    main()
