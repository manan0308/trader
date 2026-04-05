from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader_system.llm.critic_flow import merge_critic_review, overlay_materiality


def test_overlay_materiality_triggers_on_cash_shift() -> None:
    base = {"NIFTY": 0.20, "US": 0.20, "GOLD": 0.20, "CASH": 0.40}
    candidate = {"NIFTY": 0.17, "US": 0.18, "GOLD": 0.15, "CASH": 0.50}

    materiality = overlay_materiality(base, candidate)

    assert materiality["is_material"] is True
    assert "cash_delta" in materiality["reasons"]


def test_merge_critic_review_reduces_risk_and_zeroes_bias_on_direction_disagree() -> None:
    overlay = {
        "default_risk_off_override": 0.0,
        "dates": [
            {
                "date": "2026-04-02",
                "holding_days": 7,
                "risk_off_override": 0.35,
                "asset_bias": {"NIFTY": -0.2, "MIDCAP": -0.4, "SMALLCAP": -0.5, "GOLD": 0.4, "SILVER": 0.1, "US": 0.3},
                "rationale": "event risk",
                "confidence": 0.8,
            }
        ],
    }
    critic = {
        "review": {
            "verdict": "disagree",
            "confidence": 0.80,
            "main_reason": "Weak evidence.",
            "issue_type": "direction",
            "historical_support": "weakens",
            "suggested_risk_adjustment": -0.05,
            "suggested_holding_days": 3,
            "watch_items": [],
        }
    }

    merged, meta = merge_critic_review(overlay, critic)

    assert meta["critic_used"] is True
    assert merged["dates"][0]["risk_off_override"] == 0.10
    assert merged["dates"][0]["holding_days"] == 3
    assert all(value == 0.0 for value in merged["dates"][0]["asset_bias"].values())
