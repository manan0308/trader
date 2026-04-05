from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from strategy.v9_engine import ALL, RISKY
from llm.anthropic_risk_overlay import packet_event_risk
from runtime.audit_log import write_audit_runs
from runtime.store import AUDIT_RUNS_PATH, LEARNING_STATE_PATH, read_json, read_jsonl, write_json


HORIZONS = [1, 3, 5]
EWMA_ALPHA = 0.12


def empty_stat() -> Dict[str, Any]:
    return {
        "count": 0,
        "ewma_delta": 0.0,
        "avg_delta": 0.0,
        "ewma_win_rate": 0.5,
    }


def default_learning_state() -> Dict[str, Any]:
    overlay_block = {
        "trust_multiplier": 1.0,
        "max_risk_off_cap": 0.35,
        "allow_asset_bias": True,
        "min_event_risk_to_call": 0.70,
        "cooldown_days_remaining": 0,
        "global": {f"{h}d": empty_stat() for h in HORIZONS},
        "contexts": {"stress": {f"{h}d": empty_stat() for h in HORIZONS}, "calm": {f"{h}d": empty_stat() for h in HORIZONS}},
    }
    return {
        "updated_at": None,
        "overlay": overlay_block,
        "overlay_policy": dict(overlay_block),
        "model_governance": {
            "status": "warming_up",
            "reason": "not_enough_completed_runs",
            "trailing_21d": {},
            "trailing_63d": {},
            "manual_review_required": False,
        },
    }


def load_learning_state() -> Dict[str, Any]:
    state = read_json(LEARNING_STATE_PATH, default=None)
    if not isinstance(state, dict):
        return default_learning_state()
    if "overlay" not in state or not isinstance(state.get("overlay"), dict):
        state["overlay"] = default_learning_state()["overlay"]
    if "overlay_policy" not in state and "overlay" in state:
        state["overlay_policy"] = state["overlay"]
    if "model_governance" not in state or not isinstance(state.get("model_governance"), dict):
        state["model_governance"] = default_learning_state()["model_governance"]
    for key, value in default_learning_state()["overlay"].items():
        state["overlay"].setdefault(key, value)
        state["overlay_policy"].setdefault(key, value)
    return state


def weights_from_record(record: Mapping[str, Any], key: str) -> pd.Series:
    raw = record.get(key, {}) if isinstance(record, Mapping) else {}
    series = pd.Series(0.0, index=ALL)
    if isinstance(raw, dict):
        for asset in ALL:
            if asset in raw:
                series[asset] = float(raw[asset])
    total = float(series.sum())
    if total > 0:
        series = series / total
    else:
        series["CASH"] = 1.0
    return series


def horizon_return(prices: pd.DataFrame, signal_date: str, weights: pd.Series, horizon: int) -> float | None:
    if signal_date not in prices.index.strftime("%Y-%m-%d"):
        return None
    index_lookup = {dt.strftime("%Y-%m-%d"): i for i, dt in enumerate(prices.index)}
    loc = index_lookup.get(signal_date)
    if loc is None or loc + horizon >= len(prices.index):
        return None
    daily = prices.pct_change(fill_method=None).fillna(0.0).iloc[loc + 1 : loc + horizon + 1]
    series = (daily[ALL] * weights).sum(axis=1)
    return float((1.0 + series).prod() - 1.0)


def current_context_label(record: Mapping[str, Any]) -> str:
    macro_state = record.get("macro_state", {})
    if not isinstance(macro_state, dict):
        macro_state = record.get("packet_snapshot", {}).get("macro_state", {})
    if not isinstance(macro_state, dict):
        return "calm"
    triggered, _ = packet_event_risk({"macro_state": macro_state})
    return "stress" if triggered else "calm"


def finalize_audit_evaluations(runs: List[Dict[str, Any]], benchmark_prices: pd.DataFrame) -> List[Dict[str, Any]]:
    for run in runs:
        evals = run.setdefault("evaluations", {})
        signal_date = str(run.get("as_of") or run.get("signal_as_of") or "")
        base_weights = weights_from_record(run, "v9_base_weights")
        final_weights = weights_from_record(run, "v9_final_weights")
        eqwt_weights = pd.Series(0.0, index=ALL)
        for asset in RISKY:
            eqwt_weights[asset] = 1.0 / len(RISKY)

        for horizon in HORIZONS:
            key = f"{horizon}d"
            if isinstance(evals.get(key), dict):
                continue
            base_ret = horizon_return(benchmark_prices, signal_date, base_weights, horizon)
            final_ret = horizon_return(benchmark_prices, signal_date, final_weights, horizon)
            eqwt_ret = horizon_return(benchmark_prices, signal_date, eqwt_weights, horizon)
            if base_ret is None or final_ret is None or eqwt_ret is None:
                continue
            evals[key] = {
                "base_return": base_ret,
                "final_return": final_ret,
                "eqwt_return": eqwt_ret,
                "overlay_delta": final_ret - base_ret,
            }
    return runs


def update_stat(stat: Dict[str, Any], delta: float) -> Dict[str, Any]:
    count = int(stat.get("count", 0)) + 1
    prev_avg = float(stat.get("avg_delta", 0.0))
    prev_ewma = float(stat.get("ewma_delta", 0.0))
    prev_win = float(stat.get("ewma_win_rate", 0.5))

    stat["count"] = count
    stat["avg_delta"] = prev_avg + (delta - prev_avg) / count
    stat["ewma_delta"] = (1.0 - EWMA_ALPHA) * prev_ewma + EWMA_ALPHA * delta
    stat["ewma_win_rate"] = (1.0 - EWMA_ALPHA) * prev_win + EWMA_ALPHA * (1.0 if delta > 0 else 0.0)
    return stat


def trust_from_stats(global_stats: Mapping[str, Any], context_stats: Mapping[str, Any]) -> float:
    score_5 = float(global_stats.get("5d", {}).get("ewma_delta", 0.0))
    win_5 = float(global_stats.get("5d", {}).get("ewma_win_rate", 0.5))
    count_5 = int(global_stats.get("5d", {}).get("count", 0))
    context_score_5 = float(context_stats.get("5d", {}).get("ewma_delta", 0.0))
    context_count_5 = int(context_stats.get("5d", {}).get("count", 0))

    trust = 1.0
    if count_5 >= 12:
        if score_5 < -0.004 or win_5 < 0.45:
            trust = 0.50
        elif score_5 < -0.0015:
            trust = 0.75
    if context_count_5 >= 6:
        if context_score_5 < -0.004:
            trust = min(trust, 0.50)
        elif context_score_5 < -0.0015:
            trust = min(trust, 0.75)
    return trust


def cooldown_days(runs: List[Dict[str, Any]]) -> int:
    overlay_runs = [run for run in runs if bool(run.get("overlay", {}).get("raw_active", False))]
    recent = overlay_runs[-10:]
    bad = 0
    for run in recent:
        row = run.get("evaluations", {}).get("5d")
        if isinstance(row, dict) and float(row.get("overlay_delta", 0.0)) < -0.003:
            bad += 1
    return 5 if bad >= 3 else 0


def governance_from_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [run for run in runs if isinstance(run.get("evaluations", {}).get("1d"), dict)]
    if len(completed) < 15:
        return {
            "status": "warming_up",
            "reason": "not_enough_completed_runs",
            "trailing_21d": {},
            "trailing_63d": {},
            "manual_review_required": False,
        }

    final_1d = [float(run["evaluations"]["1d"]["final_return"]) for run in completed]
    eqwt_1d = [float(run["evaluations"]["1d"]["eqwt_return"]) for run in completed]

    def trailing_window(window: int) -> Dict[str, float]:
        lhs = pd.Series(final_1d[-window:] if len(final_1d) >= window else final_1d)
        rhs = pd.Series(eqwt_1d[-window:] if len(eqwt_1d) >= window else eqwt_1d)
        final_cum = float((1.0 + lhs).prod() - 1.0)
        eqwt_cum = float((1.0 + rhs).prod() - 1.0)
        equity = (1.0 + lhs).cumprod()
        max_dd = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0
        return {
            "final_return": final_cum,
            "eqwt_return": eqwt_cum,
            "relative_return": final_cum - eqwt_cum,
            "max_drawdown": max_dd,
        }

    trailing21 = trailing_window(21)
    trailing63 = trailing_window(63)
    review = trailing63["relative_return"] < -0.04 or (trailing21["relative_return"] < -0.02 and trailing21["max_drawdown"] < -0.08)
    status = "review" if review else "normal"
    reason = "relative_underperformance" if review else "within_guardrails"
    return {
        "status": status,
        "reason": reason,
        "trailing_21d": trailing21,
        "trailing_63d": trailing63,
        "manual_review_required": review,
    }


def build_learning_state(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = default_learning_state()
    overlay_global = state["overlay"]["global"]
    overlay_contexts = state["overlay"]["contexts"]

    for run in runs:
        if not bool(run.get("overlay", {}).get("raw_active", False)):
            continue
        context = current_context_label(run)
        evals = run.get("evaluations", {})
        for horizon in HORIZONS:
            key = f"{horizon}d"
            row = evals.get(key)
            if not isinstance(row, dict):
                continue
            delta = float(row.get("overlay_delta", 0.0))
            update_stat(overlay_global[key], delta)
            update_stat(overlay_contexts[context][key], delta)

    latest_context = current_context_label(runs[-1]) if runs else "calm"
    trust = trust_from_stats(overlay_global, overlay_contexts[latest_context])
    calm_5d = overlay_contexts["calm"]["5d"]
    state["overlay"]["trust_multiplier"] = trust
    state["overlay"]["allow_asset_bias"] = trust >= 0.75
    state["overlay"]["max_risk_off_cap"] = 0.20 if int(calm_5d.get("count", 0)) >= 6 and float(calm_5d.get("ewma_delta", 0.0)) < -0.001 else 0.35
    state["overlay"]["cooldown_days_remaining"] = cooldown_days(runs)
    state["overlay_policy"] = state["overlay"]
    state["model_governance"] = governance_from_runs(runs)
    state["updated_at"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    return state


def apply_learning_to_overlay(raw_overlay: Mapping[str, Any], learning_state: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    policy = learning_state.get("overlay_policy", learning_state.get("overlay", {}))
    multiplier = float(policy.get("trust_multiplier", 1.0))
    multiplier = float(np.clip(multiplier, 0.5, 1.0))
    max_risk_cap = float(np.clip(float(policy.get("max_risk_off_cap", 0.35)), 0.0, 0.6))
    allow_asset_bias = bool(policy.get("allow_asset_bias", True))
    cooldown = int(policy.get("cooldown_days_remaining", 0))

    if cooldown > 0:
        effective = {"default_risk_off_override": 0.0, "dates": []}
        applied = {
            "trust_multiplier": multiplier,
            "max_risk_off_cap": max_risk_cap,
            "allow_asset_bias": allow_asset_bias,
            "cooldown_days_remaining": cooldown,
            "raw_active": bool(float(raw_overlay.get("default_risk_off_override", 0.0)) > 0 or raw_overlay.get("dates")),
            "effective_active": False,
        }
        return effective, applied

    effective = {
        "default_risk_off_override": float(np.clip(float(raw_overlay.get("default_risk_off_override", 0.0)) * multiplier, 0.0, max_risk_cap)),
        "dates": [],
    }

    for row in raw_overlay.get("dates", []) or []:
        if not isinstance(row, dict):
            continue
        bias = row.get("asset_bias", {}) if isinstance(row.get("asset_bias", {}), dict) else {}
        effective["dates"].append(
            {
                "date": row.get("date"),
                "holding_days": int(row.get("holding_days", 1)),
                "risk_off_override": float(np.clip(float(row.get("risk_off_override", 0.0)) * multiplier, 0.0, max_risk_cap)),
                "asset_bias": {
                    asset: (
                        float(np.clip(float(bias.get(asset, 0.0)) * multiplier, -1.0, 1.0))
                        if allow_asset_bias
                        else 0.0
                    )
                    for asset in RISKY
                },
                "rationale": row.get("rationale", ""),
                "confidence": row.get("confidence", 0.0),
            }
        )

    applied = {
        "trust_multiplier": multiplier,
        "max_risk_off_cap": max_risk_cap,
        "allow_asset_bias": allow_asset_bias,
        "cooldown_days_remaining": cooldown,
        "raw_active": bool(float(raw_overlay.get("default_risk_off_override", 0.0)) > 0 or effective["dates"]),
        "effective_active": bool(float(effective.get("default_risk_off_override", 0.0)) > 0 or effective["dates"]),
    }
    return effective, applied


def load_audit_runs() -> List[Dict[str, Any]]:
    return read_jsonl(AUDIT_RUNS_PATH)


def save_audit_runs(runs: List[Dict[str, Any]]) -> None:
    write_audit_runs(runs)


def update_learning_outputs(benchmark_prices: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    runs = load_audit_runs()
    runs = finalize_audit_evaluations(runs, benchmark_prices)
    state = build_learning_state(runs)
    save_audit_runs(runs)
    write_json(LEARNING_STATE_PATH, state)
    return runs, state


def apply_overlay_learning_policy(raw_overlay: Mapping[str, Any], learning_state: Mapping[str, Any]) -> Dict[str, Any]:
    effective, _ = apply_learning_to_overlay(raw_overlay, learning_state)
    return effective


def update_learning_state(
    benchmark_prices: pd.DataFrame,
    audit_runs: List[Dict[str, Any]],
    state: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    runs = finalize_audit_evaluations(audit_runs, benchmark_prices)
    new_state = build_learning_state(runs)
    if state and isinstance(state, Mapping):
        new_state["prior_overlay_policy"] = state.get("overlay_policy", {})
    save_audit_runs(runs)
    write_json(LEARNING_STATE_PATH, new_state)
    return new_state
