from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from trader_system.strategy.v9_engine import ALL, RISKY, NarrativeOverlay, StrategyConfig, run_strategy


def overlay_from_payload(payload: Mapping[str, Any] | None, index: pd.DatetimeIndex) -> NarrativeOverlay | None:
    if not payload:
        return None

    risk_off = pd.Series(0.0, index=index)
    asset_bias = pd.DataFrame(0.0, index=index, columns=RISKY)
    default_risk = float(np.clip(float(payload.get("default_risk_off_override", 0.0) or 0.0), 0.0, 1.0))
    risk_off[:] = default_risk

    for item in payload.get("dates", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            start = pd.Timestamp(item.get("date"))
        except Exception:
            continue
        active = index[index >= start]
        if len(active) == 0:
            continue
        holding_days = max(1, int(item.get("holding_days", 1)))
        window = active[:holding_days]
        risk_value = float(np.clip(float(item.get("risk_off_override", 0.0)), 0.0, 1.0))
        risk_off.loc[window] = np.maximum(risk_off.loc[window], risk_value)
        bias_map = item.get("asset_bias", {}) if isinstance(item.get("asset_bias"), Mapping) else {}
        for asset in RISKY:
            if asset in bias_map:
                asset_bias.loc[window, asset] = float(np.clip(float(bias_map[asset]), -1.0, 1.0))

    return NarrativeOverlay(risk_off=risk_off.clip(lower=0.0, upper=1.0), asset_bias=asset_bias.clip(lower=-1.0, upper=1.0))


def latest_weights_from_overlay_payload(
    prices: pd.DataFrame,
    config: StrategyConfig,
    overlay_payload: Mapping[str, Any] | None,
) -> Dict[str, float]:
    overlay = overlay_from_payload(overlay_payload, prices.index)
    weights = run_strategy(prices, config, overlay=overlay)
    return {asset: float(weights.iloc[-1][asset]) for asset in weights.columns}


def overlay_materiality(base_weights: Mapping[str, Any], candidate_weights: Mapping[str, Any]) -> Dict[str, Any]:
    assets = sorted(set(base_weights) | set(candidate_weights) | set(ALL))
    base = pd.Series({asset: float(base_weights.get(asset, 0.0)) for asset in assets}, dtype=float)
    cand = pd.Series({asset: float(candidate_weights.get(asset, 0.0)) for asset in assets}, dtype=float)
    delta = cand - base
    abs_delta = delta.abs()

    cash_delta = float(abs_delta.get("CASH", 0.0))
    risky_delta = float(abs_delta[[asset for asset in assets if asset != "CASH"]].sum() / 2.0)
    max_abs_asset_delta = float(abs_delta.max()) if len(abs_delta) else 0.0
    changed_assets = [asset for asset in assets if abs(float(delta.get(asset, 0.0))) >= 0.01]
    reasons = []
    if cash_delta >= 0.03:
        reasons.append("cash_delta")
    if max_abs_asset_delta >= 0.025:
        reasons.append("single_asset_delta")
    if risky_delta >= 0.05:
        reasons.append("half_turnover")
    return {
        "is_material": bool(reasons),
        "cash_delta": cash_delta,
        "half_turnover": risky_delta,
        "max_abs_asset_delta": max_abs_asset_delta,
        "changed_assets": changed_assets,
        "reasons": reasons,
        "delta": {asset: float(delta[asset]) for asset in assets},
    }


def merge_critic_review(policy_overlay: Mapping[str, Any], critic_payload: Mapping[str, Any] | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    final_overlay = deepcopy(dict(policy_overlay))
    review = (critic_payload or {}).get("review", critic_payload or {})
    if not isinstance(review, Mapping):
        return final_overlay, {"critic_used": False, "reason": "missing_review"}

    verdict = str(review.get("verdict", "agree"))
    confidence = float(np.clip(float(review.get("confidence", 0.0)), 0.0, 1.0))
    issue_type = str(review.get("issue_type", "none"))
    suggested_holding_days = int(np.clip(int(review.get("suggested_holding_days", 5)), 1, 10))
    raw_adjustment = float(review.get("suggested_risk_adjustment", 0.0))
    risk_adjustment = min(raw_adjustment, 0.0)

    scale = {"agree": 0.0, "partially_agree": 0.5, "disagree": 1.0}.get(verdict, 0.0)
    applied_adjustment = risk_adjustment * scale

    final_overlay["default_risk_off_override"] = float(
        np.clip(float(final_overlay.get("default_risk_off_override", 0.0)) + applied_adjustment, 0.0, 0.6)
    )

    zero_bias = issue_type in {"direction", "weak_evidence"} and verdict in {"partially_agree", "disagree"}
    halve_bias = issue_type in {"direction", "weak_evidence"} and verdict == "partially_agree"

    adjusted_rows = []
    for row in final_overlay.get("dates", []) or []:
        if not isinstance(row, Mapping):
            continue
        updated = dict(row)
        updated["risk_off_override"] = float(np.clip(float(updated.get("risk_off_override", 0.0)) + applied_adjustment, 0.0, 0.6))
        updated["holding_days"] = min(int(updated.get("holding_days", 1)), suggested_holding_days)
        bias = dict(updated.get("asset_bias", {}) or {})
        if zero_bias:
            bias = {asset: 0.0 for asset in RISKY}
        elif halve_bias:
            bias = {asset: float(np.clip(float(bias.get(asset, 0.0)) * 0.5, -1.0, 1.0)) for asset in RISKY}
        updated["asset_bias"] = {asset: float(np.clip(float(bias.get(asset, 0.0)), -1.0, 1.0)) for asset in RISKY}
        adjusted_rows.append(updated)

    if verdict == "disagree" and confidence >= 0.75 and issue_type in {"direction", "weak_evidence"}:
        final_overlay["default_risk_off_override"] = float(min(final_overlay["default_risk_off_override"], 0.10))
        for row in adjusted_rows:
            row["risk_off_override"] = float(min(float(row.get("risk_off_override", 0.0)), 0.10))
            row["asset_bias"] = {asset: 0.0 for asset in RISKY}

    final_overlay["dates"] = adjusted_rows
    meta = {
        "critic_used": True,
        "verdict": verdict,
        "confidence": confidence,
        "issue_type": issue_type,
        "applied_risk_adjustment": applied_adjustment,
        "suggested_holding_days": suggested_holding_days,
    }
    return final_overlay, meta
