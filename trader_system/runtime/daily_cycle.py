#!/usr/bin/env python3
"""
Daily Cycle Runner
==================

One command to run the production-facing daily workflow:

1. Export the latest benchmark-mode LLM context packet.
2. Optionally call Anthropic for a sparse risk override.
3. Apply conservative learning-based trust caps to that override.
4. Refresh recent live-signal snapshots.
5. Build a tradable dry-run execution plan.
6. Update paper-trading state, audit logs, and daily learning state.
7. Sync the React dashboard data.

This does not place broker orders.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from trader_system.runtime.audit_log import AUDIT_JSONL_PATH, AUDIT_LATEST_PATH, stable_hash, upsert_audit_run
from trader_system.strategy.v9_engine import (
    CACHE_DIR,
    DEFAULT_BACKTEST_START,
    DEFAULT_TX,
    YahooFinanceSource,
    load_llm_overlay,
    run_strategy,
    schedule_flags,
)
from research.alpha_v11_macro_value_research import fetch_macro_panel
from research.alpha_v12_meta_ensemble import (
    BASE_V9,
    LLM_PACKET_PATH,
    build_latest_llm_packet,
    build_meta_inputs,
    build_sleeves,
    current_meta_target,
    meta_candidates,
    run_meta_strategy,
)
from trader_system.llm.overlay_learning import (
    LEARNING_STATE_PATH,
    apply_learning_to_overlay,
    load_learning_state,
    update_learning_state,
)
from trader_system.llm.critic_flow import (
    latest_weights_from_overlay_payload,
    merge_critic_review,
    overlay_materiality,
)
from trader_system.events.structured_event_store import (
    ACTIVE_EVENTS_PATH,
    LATEST_EVENTS_PATH,
    attach_structured_context,
    refresh_structured_event_store,
)
from trader_system.broker.groww_client import PRODUCTION_GROWW_UNIVERSE_PATH
from trader_system.runtime.india_market_calendar import market_clock
from trader_system.runtime.paper_ledger import (
    BASELINE_PAPER_PATHS,
    DEFAULT_PAPER_PATHS,
    PAPER_JOURNAL_PATH,
    PAPER_LATEST_PATH,
    PAPER_STATE_PATH,
    update_paper_account_from_target,
)
from trader_system.runtime.env_loader import load_runtime_env
from trader_system.runtime.store import (
    EXECUTION_PLAN_BASE_PATH,
    PAPER_BASE_HISTORY_PATH,
    PAPER_BASE_STATE_PATH,
    PAPER_BASE_SUMMARY_PATH,
    PAPER_COMPARISON_PATH,
    load_json,
    write_json,
)


BASE_DIR = Path(__file__).resolve().parents[2]
RAW_OVERLAY_PATH = CACHE_DIR / "anthropic_overlay_latest.json"
POLICY_OVERLAY_PATH = CACHE_DIR / "policy_overlay_latest.json"
ACTIVE_OVERLAY_PATH = CACHE_DIR / "active_overlay_latest.json"
CRITIC_PACKET_PATH = CACHE_DIR / "llm_review_packet_latest.json"
CRITIC_REVIEW_PATH = CACHE_DIR / "llm_disagreement_review_latest.json"
PATTERN_LAB_PATH = CACHE_DIR / "pattern_signal_lab.json"
LATEST_RUN_PATH = CACHE_DIR / "daily_cycle_latest.json"


def run_cmd(cmd: List[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def latest_model_observations(live_payload: dict) -> dict:
    observations = {}
    for model, info in (live_payload.get("models", {}) or {}).items():
        resolved = [row for row in info.get("rows", []) if row.get("next_day_return") is not None]
        if not resolved:
            continue
        row = resolved[-1]
        observations[model] = {
            "signal_date": row.get("signal_date"),
            "action_date": row.get("action_date"),
            "next_day_return": float(row["next_day_return"]),
        }
    return observations


def overlay_is_active(payload: dict) -> bool:
    if float(payload.get("default_risk_off_override", 0.0) or 0.0) > 0:
        return True
    return bool(payload.get("dates"))


def latest_v9_weights(prices: pd.DataFrame, overlay_path: Path | None) -> Dict[str, float]:
    overlay = load_llm_overlay(str(overlay_path), prices.index) if overlay_path and overlay_path.exists() else None
    weights = run_strategy(prices, BASE_V9, overlay=overlay)
    return {asset: float(weights.iloc[-1][asset]) for asset in weights.columns}


def export_llm_packet(
    start: str,
    refresh: bool,
    packet_path: Path,
    overlay_path: Path | None = None,
) -> pd.DataFrame:
    prices = YahooFinanceSource(universe_mode="benchmark").fetch(start, refresh=refresh)
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"), refresh=refresh)
    overlay = load_llm_overlay(str(overlay_path), prices.index) if overlay_path and overlay_path.exists() else None

    config = meta_candidates()[3]
    asset_weights = run_meta_strategy(prices, macro, config, overlay=overlay, tx_cost=DEFAULT_TX)
    sleeves = build_sleeves(prices, macro, overlay=overlay)
    meta_inputs = build_meta_inputs(prices, macro, sleeves, tx_cost=DEFAULT_TX)
    sleeve_returns = meta_inputs["returns"]  # type: ignore[assignment]
    breadth = meta_inputs["breadth"]  # type: ignore[assignment]
    vix_ratio = meta_inputs["vix_ratio"]  # type: ignore[assignment]

    risk_off = (
        overlay.risk_off.reindex(sleeve_returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if overlay is not None
        else pd.Series(0.0, index=sleeve_returns.index)
    )
    schedule = schedule_flags(sleeve_returns.index, "MONTHLY")
    current = pd.Series({"composite": 0.50, "v9": 0.30, "macro": 0.20, "canary": 0.0}, dtype=float)
    meta_history = pd.DataFrame(0.0, index=sleeve_returns.index, columns=current.index)

    min_history = max(config.sleeve_mom_slow, 252)
    for i, dt in enumerate(sleeve_returns.index):
        if i >= min_history and bool(schedule.loc[dt]):
            target = current_meta_target(
                dt=dt,
                subset=sleeve_returns.iloc[: i + 1],
                breadth=breadth,
                vix_ratio=vix_ratio,
                config=config,
                llm_risk=float(risk_off.loc[dt]),
            )
            if float((target - current).abs().max()) > config.trade_band:
                current = config.trade_mix * target + (1.0 - config.trade_mix) * current
                current = current / current.sum()
        meta_history.loc[dt] = current

    build_latest_llm_packet(
        prices=prices,
        macro=macro,
        meta_weights=meta_history,
        asset_weights=asset_weights,
        config=config,
        path=packet_path,
    )
    return prices


def ensure_noop_overlay(path: Path) -> dict:
    payload = {"default_risk_off_override": 0.0, "dates": []}
    write_json(path, payload)
    return payload


def overlay_env_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def enrich_packet_context(packet_path: Path, *, refresh: bool = False) -> tuple[dict, dict]:
    packet_payload = load_json(packet_path, default={})
    as_of = packet_payload.get("as_of")
    event_store = refresh_structured_event_store(as_of=as_of, refresh=refresh)
    pattern_lab = load_json(PATTERN_LAB_PATH, default={})
    enriched = attach_structured_context(packet_payload, event_store=event_store, pattern_lab=pattern_lab)
    write_json(packet_path, enriched)
    return enriched, event_store


def find_node_binary() -> str | None:
    candidates = [
        shutil.which("node"),
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def run_execution_plan(
    *,
    execution_start: str,
    execution_data_source: str,
    llm_override_path: Path | None,
    cash: float,
    portfolio_file: str | None,
    groww_live: bool,
    groww_universe_file: str | None,
    refresh_cache: bool,
    output_path: Path,
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "trader_system.execution.order_planner",
        "--data-source",
        execution_data_source,
        "--universe-mode",
        "tradable",
        "--start",
        execution_start,
        "--cash",
        str(cash),
        "--output",
        str(output_path),
    ]
    if llm_override_path:
        cmd.extend(["--llm-override-file", str(llm_override_path)])
    if portfolio_file:
        cmd.extend(["--portfolio-file", portfolio_file])
    if groww_live:
        cmd.append("--groww-live")
    if groww_universe_file:
        cmd.extend(["--groww-universe-file", groww_universe_file])
    if refresh_cache:
        cmd.append("--refresh-cache")
    run_cmd(cmd, cwd=BASE_DIR)
    return load_json(output_path, default={})


def main() -> None:
    load_runtime_env(override=True)
    parser = argparse.ArgumentParser(description="Run the production-facing daily signal cycle.")
    parser.add_argument("--benchmark-start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--execution-start", default="2024-01-01")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--overlay-model", default="claude-opus-4-6")
    parser.add_argument("--overlay-call-mode", choices=["event", "always"], default="always")
    parser.add_argument("--overlay-web-search", dest="overlay_web_search", action="store_true")
    parser.add_argument("--no-overlay-web-search", dest="overlay_web_search", action="store_false")
    parser.add_argument("--overlay-web-search-uses", type=int, default=2)
    parser.add_argument("--llm-override-file")
    parser.add_argument("--portfolio-file")
    parser.add_argument("--groww-live", action="store_true")
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--groww-universe-file")
    parser.add_argument("--execution-data-source", choices=["yfinance", "groww"], default="yfinance")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--paper-initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--reset-paper", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--build-dashboard", action="store_true")
    parser.set_defaults(overlay_web_search=True)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    packet_path = Path(LLM_PACKET_PATH)
    source_overlay_path = Path(args.llm_override_file).expanduser().resolve() if args.llm_override_file else RAW_OVERLAY_PATH

    if args.reset_state:
        for path in [
            PAPER_STATE_PATH,
            PAPER_LATEST_PATH,
            PAPER_JOURNAL_PATH,
            PAPER_BASE_STATE_PATH,
            PAPER_BASE_SUMMARY_PATH,
            PAPER_BASE_HISTORY_PATH,
            LEARNING_STATE_PATH,
            AUDIT_JSONL_PATH,
            AUDIT_LATEST_PATH,
            ACTIVE_OVERLAY_PATH,
            POLICY_OVERLAY_PATH,
            CRITIC_PACKET_PATH,
            CRITIC_REVIEW_PATH,
            EXECUTION_PLAN_BASE_PATH,
            PAPER_COMPARISON_PATH,
            LATEST_RUN_PATH,
        ]:
            if path.exists():
                path.unlink()
    elif args.reset_paper:
        for path in [
            PAPER_STATE_PATH,
            PAPER_LATEST_PATH,
            PAPER_JOURNAL_PATH,
            PAPER_BASE_STATE_PATH,
            PAPER_BASE_SUMMARY_PATH,
            PAPER_BASE_HISTORY_PATH,
            PAPER_COMPARISON_PATH,
        ]:
            if path.exists():
                path.unlink()

    prior_learning = load_learning_state()
    prior_overlay_policy = prior_learning.get("overlay_policy", {})

    packet_overlay_path = None
    if not args.skip_overlay:
        if ACTIVE_OVERLAY_PATH.exists():
            packet_overlay_path = ACTIVE_OVERLAY_PATH
        elif source_overlay_path.exists():
            packet_overlay_path = source_overlay_path
    benchmark_prices = export_llm_packet(
        start=args.benchmark_start,
        refresh=args.refresh_cache,
        packet_path=packet_path,
        overlay_path=packet_overlay_path,
    )
    packet_payload, event_store = enrich_packet_context(packet_path, refresh=args.refresh_cache)
    clock = market_clock()
    latest_bar_day = benchmark_prices.index[-1].date().isoformat()
    skip_execution = latest_bar_day != clock.today
    skip_reason = None
    if skip_execution:
        if not clock.is_trading_day:
            skip_reason = f"market_closed:{clock.session}:{clock.holiday_name or 'non_trading_day'}"
        else:
            skip_reason = f"latest_bar_stale:{latest_bar_day}"

    if args.skip_overlay:
        raw_overlay = ensure_noop_overlay(source_overlay_path)
    elif args.llm_override_file:
        raw_overlay = load_json(source_overlay_path, default={"default_risk_off_override": 0.0, "dates": []})
    elif int(prior_overlay_policy.get("cooldown_days_remaining", 0)) > 0:
        raw_overlay = ensure_noop_overlay(source_overlay_path)
    elif not overlay_env_available():
        raw_overlay = ensure_noop_overlay(source_overlay_path)
    else:
        cmd = [
            sys.executable,
            "-m",
            "trader_system.llm.anthropic_risk_overlay",
            "--packet",
            str(packet_path),
            "--output",
            str(source_overlay_path),
            "--model",
            args.overlay_model,
            "--call-mode",
            args.overlay_call_mode,
        ]
        if args.overlay_web_search:
            cmd.extend(["--enable-web-search", "--web-search-uses", str(args.overlay_web_search_uses)])
        try:
            run_cmd(cmd, cwd=BASE_DIR)
            raw_overlay = load_json(source_overlay_path, default={"default_risk_off_override": 0.0, "dates": []})
        except subprocess.CalledProcessError:
            raw_overlay = ensure_noop_overlay(source_overlay_path)

    policy_overlay, applied_policy = apply_learning_to_overlay(raw_overlay, prior_learning)
    write_json(POLICY_OVERLAY_PATH, policy_overlay)

    base_v9_weights = latest_weights_from_overlay_payload(benchmark_prices, BASE_V9, overlay_payload=None)
    policy_v9_weights = latest_weights_from_overlay_payload(benchmark_prices, BASE_V9, overlay_payload=policy_overlay)
    materiality = overlay_materiality(base_v9_weights, policy_v9_weights)

    critic_payload: dict = {
        "status": "skipped",
        "reason": "overlay_inactive" if not overlay_is_active(policy_overlay) else "overlay_not_material",
        "materiality": materiality,
    }
    critic_meta = {"critic_used": False, "reason": critic_payload["reason"]}
    active_overlay = policy_overlay

    if overlay_is_active(policy_overlay) and materiality["is_material"]:
        review_cmd = [
            sys.executable,
            "-m",
            "trader_system.llm.build_review_packet",
            "--start",
            args.benchmark_start,
            "--packet-file",
            str(packet_path),
            "--overlay-file",
            str(POLICY_OVERLAY_PATH),
            "--pattern-file",
            str(PATTERN_LAB_PATH),
            "--output",
            str(CRITIC_PACKET_PATH),
        ]
        try:
            run_cmd(review_cmd, cwd=BASE_DIR)
            critic_cmd = [
                sys.executable,
                "-m",
                "trader_system.llm.anthropic_disagreement_review",
                "--packet",
                str(CRITIC_PACKET_PATH),
                "--prompt",
                str(BASE_DIR / "config" / "prompts" / "llm_disagreement_review_prompt.md"),
                "--output",
                str(CRITIC_REVIEW_PATH),
                "--model",
                args.overlay_model,
            ]
            run_cmd(critic_cmd, cwd=BASE_DIR)
            critic_payload = load_json(CRITIC_REVIEW_PATH, default={})
            active_overlay, critic_meta = merge_critic_review(policy_overlay, critic_payload)
        except subprocess.CalledProcessError as exc:
            critic_payload = {
                "status": "error",
                "reason": f"critic_failed:{exc.returncode}",
                "materiality": materiality,
            }
        write_json(CRITIC_REVIEW_PATH, critic_payload)
    else:
        write_json(CRITIC_REVIEW_PATH, critic_payload)

    write_json(ACTIVE_OVERLAY_PATH, active_overlay)

    live_cmd = [
        sys.executable,
        "-m",
        "trader_system.runtime.live_signal_report",
        "--model",
        "all",
        "--days",
        str(args.days),
        "--universe-mode",
        "benchmark",
        "--llm-override-file",
        str(ACTIVE_OVERLAY_PATH),
    ]
    if args.refresh_cache:
        live_cmd.append("--refresh-cache")
    run_cmd(live_cmd, cwd=BASE_DIR)

    live_payload = load_json(CACHE_DIR / "live_signal_latest.json", default={})
    if (not PAPER_BASE_STATE_PATH.exists()) and PAPER_STATE_PATH.exists():
        write_json(PAPER_BASE_STATE_PATH, load_json(PAPER_STATE_PATH, default={}))
        write_json(PAPER_BASE_SUMMARY_PATH, load_json(PAPER_LATEST_PATH, default={}))
        if PAPER_JOURNAL_PATH.exists() and not PAPER_BASE_HISTORY_PATH.exists():
            PAPER_BASE_HISTORY_PATH.write_text(PAPER_JOURNAL_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    base_execution_payload = run_execution_plan(
        execution_start=args.execution_start,
        execution_data_source=args.execution_data_source,
        llm_override_path=None,
        cash=args.cash,
        portfolio_file=args.portfolio_file,
        groww_live=args.groww_live,
        groww_universe_file=args.groww_universe_file,
        refresh_cache=args.refresh_cache,
        output_path=EXECUTION_PLAN_BASE_PATH,
    )
    execution_payload = run_execution_plan(
        execution_start=args.execution_start,
        execution_data_source=args.execution_data_source,
        llm_override_path=ACTIVE_OVERLAY_PATH,
        cash=args.cash,
        portfolio_file=args.portfolio_file,
        groww_live=args.groww_live,
        groww_universe_file=args.groww_universe_file,
        refresh_cache=args.refresh_cache,
        output_path=CACHE_DIR / "execution_plan_latest.json",
    )

    if skip_execution:
        for output_path, payload in (
            (EXECUTION_PLAN_BASE_PATH, base_execution_payload),
            (CACHE_DIR / "execution_plan_latest.json", execution_payload),
        ):
            payload.setdefault("plan", {})
            payload["market_closed_skip"] = True
            payload["skip_reason"] = skip_reason
            payload["next_trading_day"] = clock.next_trading_day
            payload["plan"]["orders"] = []
            write_json(output_path, payload)

    final_v9_weights = latest_weights_from_overlay_payload(benchmark_prices, BASE_V9, overlay_payload=active_overlay)
    tradable_prices = YahooFinanceSource(universe_mode="tradable").fetch(args.execution_start, refresh=False)
    if skip_execution:
        paper_summary = load_json(PAPER_LATEST_PATH, default={})
        if not paper_summary:
            paper_summary = {
                "as_of": latest_bar_day,
                "cash": args.paper_initial_cash,
                "total_equity": args.paper_initial_cash,
                "net_pnl": 0.0,
                "return_since_start": 0.0,
                "pending_orders": [],
            }
        base_paper_summary = load_json(PAPER_BASE_SUMMARY_PATH, default={})
        if not base_paper_summary:
            base_paper_summary = {
                "as_of": latest_bar_day,
                "cash": args.paper_initial_cash,
                "total_equity": args.paper_initial_cash,
                "net_pnl": 0.0,
                "return_since_start": 0.0,
                "pending_orders": [],
            }
    else:
        base_paper_summary = update_paper_account_from_target(
            target_weights=base_v9_weights,
            prices=tradable_prices,
            initial_cash=args.paper_initial_cash,
            paths=BASELINE_PAPER_PATHS,
        )
        paper_summary = update_paper_account_from_target(
            target_weights=final_v9_weights,
            prices=tradable_prices,
            initial_cash=args.paper_initial_cash,
            paths=DEFAULT_PAPER_PATHS,
        )

    plan_orders = execution_payload.get("plan", {}).get("orders", []) or []
    base_plan_orders = base_execution_payload.get("plan", {}).get("orders", []) or []
    latest_override = (active_overlay.get("dates") or [])[-1] if (active_overlay.get("dates") or []) else None
    signal_as_of = live_payload.get("as_of") or packet_payload.get("as_of")
    paper_comparison = {
        "as_of": signal_as_of,
        "base_total_equity": base_paper_summary.get("total_equity"),
        "llm_total_equity": paper_summary.get("total_equity"),
        "base_return_since_start": base_paper_summary.get("return_since_start"),
        "llm_return_since_start": paper_summary.get("return_since_start"),
        "equity_delta": float((paper_summary.get("total_equity") or 0.0) - (base_paper_summary.get("total_equity") or 0.0)),
        "return_delta": float((paper_summary.get("return_since_start") or 0.0) - (base_paper_summary.get("return_since_start") or 0.0)),
        "base_order_count": len(base_plan_orders),
        "llm_order_count": len(plan_orders),
        "weight_delta": materiality,
        "base_weights": base_v9_weights,
        "llm_weights": final_v9_weights,
    }
    write_json(PAPER_COMPARISON_PATH, paper_comparison)
    overlay_meta = {
        "raw_active": overlay_is_active(raw_overlay),
        "policy_active": overlay_is_active(policy_overlay),
        "effective_active": overlay_is_active(active_overlay),
        "trust_multiplier": float(applied_policy.get("trust_multiplier", prior_overlay_policy.get("trust_multiplier", 1.0))),
        "max_risk_off_cap": float(applied_policy.get("max_risk_off_cap", prior_overlay_policy.get("max_risk_off_cap", 0.35))),
        "allow_asset_bias": bool(applied_policy.get("allow_asset_bias", prior_overlay_policy.get("allow_asset_bias", True))),
        "cooldown_days_remaining": int(applied_policy.get("cooldown_days_remaining", prior_overlay_policy.get("cooldown_days_remaining", 0))),
        "materiality": materiality,
        "critic": critic_meta,
    }
    record = {
        "run_key": stable_hash(
            {
                "signal_as_of": signal_as_of,
                "raw_overlay": raw_overlay,
                "policy_overlay": policy_overlay,
                "active_overlay": active_overlay,
                "plan_orders": plan_orders,
            }
        )[:16],
        "ran_at": datetime.now().astimezone().isoformat(),
        "as_of": signal_as_of,
        "signal_as_of": signal_as_of,
        "status": "ok",
        "selected_model": "v9",
        "order_count": len(plan_orders),
        "paper_trading": not skip_execution,
        "market_closed_skip": skip_execution,
        "skip_reason": skip_reason,
        "macro_state": packet_payload.get("macro_state", {}),
        "packet_snapshot": packet_payload,
        "event_store_snapshot": {"summary": event_store.get("summary", {}), "top_events": (event_store.get("summary", {}) or {}).get("top_events", [])},
        "raw_overlay_snapshot": raw_overlay,
        "policy_overlay_snapshot": policy_overlay,
        "overlay_snapshot": active_overlay,
        "critic_snapshot": critic_payload,
        "overlay": overlay_meta,
        "overlay_policy_before": prior_learning.get("overlay_policy", {}),
        "latest_override": latest_override,
        "v9_base_weights": base_v9_weights,
        "v9_policy_weights": policy_v9_weights,
        "v9_final_weights": final_v9_weights,
        "execution_summary": {
            "orders": len(plan_orders),
            "portfolio_value": execution_payload.get("plan", {}).get("portfolio_value"),
            "gross_turnover_value": float(sum(float(order.get("delta_value", 0.0)) for order in plan_orders)),
            "skipped": skip_execution,
            "skip_reason": skip_reason,
        },
        "base_execution_summary": {
            "orders": len(base_plan_orders),
            "portfolio_value": base_execution_payload.get("plan", {}).get("portfolio_value"),
            "gross_turnover_value": float(sum(float(order.get("delta_value", 0.0)) for order in base_plan_orders)),
            "skipped": skip_execution,
            "skip_reason": skip_reason,
        },
        "model_observations": latest_model_observations(live_payload),
        "paper_base_summary": {
            "as_of": base_paper_summary.get("as_of"),
            "total_equity": base_paper_summary.get("total_equity"),
            "net_pnl": base_paper_summary.get("net_pnl"),
            "return_since_start": base_paper_summary.get("return_since_start"),
            "pending_orders": len(base_paper_summary.get("pending_orders", [])),
        },
        "paper_summary": {
            "as_of": paper_summary.get("as_of"),
            "total_equity": paper_summary.get("total_equity"),
            "net_pnl": paper_summary.get("net_pnl"),
            "return_since_start": paper_summary.get("return_since_start"),
            "pending_orders": len(paper_summary.get("pending_orders", [])),
        },
        "paper_comparison": paper_comparison,
    }
    base_execution_payload["decision_run_key"] = record["run_key"]
    execution_payload["decision_run_key"] = record["run_key"]
    base_execution_payload["variant"] = "base_v9"
    execution_payload["variant"] = "llm_overlay"
    write_json(EXECUTION_PLAN_BASE_PATH, base_execution_payload)
    execution_payload.setdefault("groww_universe_file", args.groww_universe_file or str(PRODUCTION_GROWW_UNIVERSE_PATH))
    write_json(CACHE_DIR / "execution_plan_latest.json", execution_payload)

    audit_runs = upsert_audit_run(record)
    learning_state = update_learning_state(benchmark_prices, audit_runs, state=prior_learning)

    node_bin = find_node_binary()
    if node_bin:
        run_cmd([node_bin, str(BASE_DIR / "dashboard" / "scripts" / "sync-data.mjs")], cwd=BASE_DIR)
    if args.build_dashboard:
        if not shutil.which("npm"):
            raise RuntimeError("npm is required for --build-dashboard")
        run_cmd(["npm", "run", "build"], cwd=BASE_DIR / "dashboard")

    latest = {
        "ran_at": datetime.now().astimezone().isoformat(),
        "run_key": record["run_key"],
        "as_of": signal_as_of,
        "market_clock": market_clock().__dict__,
        "packet_path": str(packet_path),
        "structured_event_store_path": str(LATEST_EVENTS_PATH),
        "structured_event_active_path": str(ACTIVE_EVENTS_PATH),
        "raw_overlay_path": str(source_overlay_path),
        "policy_overlay_path": str(POLICY_OVERLAY_PATH),
        "active_overlay_path": str(ACTIVE_OVERLAY_PATH),
        "critic_packet_path": str(CRITIC_PACKET_PATH),
        "critic_review_path": str(CRITIC_REVIEW_PATH),
        "paper_path": str(PAPER_LATEST_PATH),
        "paper_base_path": str(PAPER_BASE_SUMMARY_PATH),
        "paper_comparison_path": str(PAPER_COMPARISON_PATH),
        "execution_plan_base_path": str(EXECUTION_PLAN_BASE_PATH),
        "audit_path": str(AUDIT_LATEST_PATH),
        "dashboard_path": str(BASE_DIR / "dashboard" / "public" / "data" / "dashboard.json"),
        "build_dashboard": bool(args.build_dashboard),
        "execution_data_source": args.execution_data_source,
        "groww_live": bool(args.groww_live),
        "market_closed_skip": skip_execution,
        "skip_reason": skip_reason,
        "paper_total_equity": paper_summary.get("total_equity"),
        "learning_trust_multiplier": learning_state.get("overlay_policy", {}).get("trust_multiplier"),
        "overlay_cooldown_days_remaining": learning_state.get("overlay_policy", {}).get("cooldown_days_remaining"),
        "overlay_active": overlay_meta["effective_active"],
        "critic_used": bool(critic_meta.get("critic_used", False)),
    }
    write_json(LATEST_RUN_PATH, latest)
    print(json.dumps(latest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "ran_at": datetime.now().astimezone().isoformat(),
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(LATEST_RUN_PATH, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise
