#!/usr/bin/env python3
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MODEL_NAME = "v21_mom10_x_vr20_top2_weekly_s050_b020_cloudsafe"
RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]
WARMUP_DAYS = 252
INITIAL_PAPER_CASH_INR = 1_000_000.0
TX_COST_RATE = 0.003
RUN_IN_MEMORY_PAPER = True
INPUT_MARKET_DATA_JSON = """{{INPUT_MARKET_DATA_JSON}}"""
INITIAL_PAPER_CASH_INR_VAR = "{{INITIAL_PAPER_CASH_INR}}"
RUN_IN_MEMORY_PAPER_VAR = "{{RUN_IN_MEMORY_PAPER}}"

# Cloud-safe input contract:
# INPUT_MARKET_DATA["rows"] must be a long-form list like:
# [
#   {"date": "2025-01-01", "asset": "NIFTY", "close": 100.0, "volume": 1000000},
#   {"date": "2025-01-01", "asset": "MIDCAP", "close": 110.0, "volume": 900000},
#   ...
# ]
# Required assets per date: NIFTY, MIDCAP, SMALLCAP, GOLD, SILVER, US
INPUT_MARKET_DATA = {
    "rows": []
}


def unresolved_template(value: str) -> bool:
    text = str(value).strip()
    return text.startswith("{{") and text.endswith("}}")


def resolve_bool(default_value: bool, raw_value: str) -> bool:
    if unresolved_template(raw_value):
        return bool(default_value)
    text = str(raw_value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default_value)


def resolve_float(default_value: float, raw_value: str) -> float:
    if unresolved_template(raw_value):
        return float(default_value)
    try:
        return float(raw_value)
    except Exception:
        return float(default_value)


def resolve_input_market_data() -> Dict[str, List[dict]]:
    if unresolved_template(INPUT_MARKET_DATA_JSON):
        return INPUT_MARKET_DATA
    try:
        payload = json.loads(INPUT_MARKET_DATA_JSON)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return INPUT_MARKET_DATA


def build_matrices(rows: List[dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not rows:
        raise RuntimeError("INPUT_MARKET_DATA['rows'] is empty.")

    frame = pd.DataFrame(rows).copy()
    required = {"date", "asset", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError("Missing input columns: " + ", ".join(sorted(missing)))

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["asset"] = frame["asset"].astype(str).str.upper()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["date", "asset", "close"])
    frame = frame[frame["asset"].isin(RISKY)].copy()

    prices = (
        frame.pivot(index="date", columns="asset", values="close")
        .sort_index()
        .reindex(columns=RISKY)
        .dropna()
    )
    volume = (
        frame.pivot(index="date", columns="asset", values="volume")
        .sort_index()
        .reindex(index=prices.index, columns=RISKY)
        .fillna(0.0)
    )
    if prices.empty:
        raise RuntimeError("No valid price matrix could be built from INPUT_MARKET_DATA.")
    return prices, volume


def schedule_flags(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    stamps = index.to_series()
    if frequency == "WEEKLY":
        bucket = stamps.dt.strftime("%Y-%U")
    elif frequency == "MONTHLY":
        bucket = stamps.dt.to_period("M").astype(str)
    else:
        raise RuntimeError("Unsupported execution frequency: " + str(frequency))
    return bucket.ne(bucket.shift(-1)).fillna(True)


def build_mom10_x_vr20_scores(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    mom10 = prices.pct_change(10, fill_method=None)
    volume_ratio20 = volume / volume.rolling(20).median().replace(0.0, np.nan)
    return mom10 * volume_ratio20


def build_top_n_targets(score_frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    targets = pd.DataFrame(0.0, index=score_frame.index, columns=ALL)
    for i, dt in enumerate(score_frame.index):
        if i < WARMUP_DAYS:
            targets.loc[dt, "CASH"] = 1.0
            continue
        score = score_frame.loc[dt].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        picks = list(score.index[:top_n])
        if not picks:
            targets.loc[dt, "CASH"] = 1.0
            continue
        weight = 1.0 / float(len(picks))
        for asset in picks:
            targets.loc[dt, asset] = weight
    return targets


def apply_rebalance_policy(targets: pd.DataFrame, frequency: str, trade_band: float, trade_step: float) -> pd.DataFrame:
    if frequency == "DAILY":
        schedule = pd.Series(True, index=targets.index)
    elif frequency == "WEEKLY":
        schedule = schedule_flags(targets.index, "WEEKLY")
    else:
        raise RuntimeError("Unsupported v21 frequency: " + str(frequency))

    weights = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    current = pd.Series(0.0, index=targets.columns)
    current["CASH"] = 1.0

    for i, dt in enumerate(targets.index):
        target = targets.loc[dt].copy()
        if i < WARMUP_DAYS:
            weights.loc[dt] = current
            continue

        if bool(schedule.loc[dt]):
            proposal = current * (1.0 - trade_step) + target * trade_step
            proposal = proposal.clip(lower=0.0)
            if float(proposal.sum()) > 0.0:
                proposal = proposal / float(proposal.sum())
            else:
                proposal = target
            if float((proposal - current).abs().max()) > trade_band:
                current = proposal
        weights.loc[dt] = current
    return weights


def format_weights_pct(mapping: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for asset in ALL:
        out[asset] = round(float(mapping.get(asset, 0.0)) * 100.0, 2)
    return out


def simulate_in_memory_paper(prices: pd.DataFrame, weights: pd.DataFrame) -> dict:
    full_prices = prices.copy()
    full_prices["CASH"] = 1.0

    initial_cash = resolve_float(INITIAL_PAPER_CASH_INR, INITIAL_PAPER_CASH_INR_VAR)
    nav = float(initial_cash)
    holdings = {asset: 0.0 for asset in ALL}
    holdings["CASH"] = nav
    history = []

    for i, dt in enumerate(full_prices.index):
        price_row = full_prices.loc[dt]

        current_values = {}
        for asset in ALL:
            if asset == "CASH":
                current_values[asset] = float(holdings["CASH"])
            else:
                current_values[asset] = float(holdings[asset]) * float(price_row[asset])
        nav = float(sum(current_values.values()))

        target = weights.loc[dt].copy()
        target_values = {asset: float(target.get(asset, 0.0)) * nav for asset in ALL}
        turnover_notional = float(sum(abs(target_values[asset] - current_values[asset]) for asset in ALL if asset != "CASH"))
        cost = turnover_notional * TX_COST_RATE * 0.5
        nav_after_cost = max(nav - cost, 0.0)

        new_holdings = {}
        for asset in ALL:
            if asset == "CASH":
                new_holdings[asset] = float(target.get(asset, 0.0)) * nav_after_cost
            else:
                px = float(price_row[asset])
                target_value = float(target.get(asset, 0.0)) * nav_after_cost
                new_holdings[asset] = target_value / px if px > 0 else 0.0
        holdings = new_holdings

        history.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "nav_inr": round(nav_after_cost, 2),
                "turnover_notional_inr": round(turnover_notional, 2),
                "cost_inr": round(cost, 2),
                "weights_pct": format_weights_pct(target.to_dict()),
            }
        )

    final_nav = history[-1]["nav_inr"] if history else float(initial_cash)
    total_return_pct = ((float(final_nav) / float(initial_cash)) - 1.0) * 100.0 if initial_cash > 0 else 0.0
    return {
        "initial_cash_inr": round(float(initial_cash), 2),
        "final_nav_inr": round(float(final_nav), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "history_tail": history[-10:],
    }


def main() -> None:
    print("Starting cloud-safe v21 signal computation...")

    resolved_input = resolve_input_market_data()
    run_paper = resolve_bool(RUN_IN_MEMORY_PAPER, RUN_IN_MEMORY_PAPER_VAR)
    prices, volume = build_matrices(resolved_input.get("rows", []))
    if len(prices) <= WARMUP_DAYS:
        raise RuntimeError("Need more history rows. This model requires more than 252 trading rows.")

    scores = build_mom10_x_vr20_scores(prices, volume)
    raw_targets = build_top_n_targets(scores, top_n=2)
    smoothed_weights = apply_rebalance_policy(raw_targets, "WEEKLY", 0.20, 0.50)

    latest_market_date = prices.index[-1].strftime("%Y-%m-%d")
    latest_raw_target = raw_targets.iloc[-1].to_dict()
    latest_live_target = smoothed_weights.iloc[-1].to_dict()
    latest_scores = scores.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "message": "Cloud-safe v21 weekly signal generated from provided in-memory market data",
        "model_name": MODEL_NAME,
        "latest_market_date": latest_market_date,
        "input_rows": int(len(resolved_input.get("rows", []))),
        "input_dates": int(len(prices.index)),
        "raw_top2_target_weights_pct": format_weights_pct(latest_raw_target),
        "smoothed_live_target_weights_pct": format_weights_pct(latest_live_target),
        "top_score_assets": {asset: round(float(value), 6) for asset, value in latest_scores.head(4).items()},
        "reference_prices": {asset: round(float(prices.iloc[-1][asset]), 4) for asset in RISKY},
        "paper_shadow": simulate_in_memory_paper(prices, smoothed_weights) if run_paper else None,
        "cloud_safe_notes": [
            "No external API calls.",
            "No environment variable reads.",
            "No file reads or writes.",
            "No live order placement.",
            "Input data must be embedded in INPUT_MARKET_DATA at the top of the script.",
        ],
    }

    print(json.dumps(payload, indent=2))
    print("Cloud-safe v21 signal computation completed successfully")


if __name__ == "__main__":
    main()
