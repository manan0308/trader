#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict, Optional

import pandas as pd

from trader_system.strategy.v9_engine import DEFAULT_TX, NarrativeOverlay, YahooFinanceSource, load_llm_overlay, run_strategy
from research.alpha_v12_meta_ensemble import BASE_V9, meta_candidates, run_meta_strategy
from research.alpha_v11_macro_value_research import fetch_macro_panel
from research.alpha_v13_sparse_meta import risk_preserving_candidates, run_sparse_meta_strategy
from research.alpha_v14_sparse_sleeves import run_sparse_sleeve_strategy, sleeve_sparse_candidates
from trader_system.runtime.india_market_calendar import holiday_name, is_trading_day, next_trading_day


def resolve_date(index: pd.DatetimeIndex, query: str) -> tuple[pd.Timestamp, bool]:
    dt = pd.Timestamp(query)
    if dt in index:
        return dt, False
    eligible = index[index <= dt]
    if len(eligible) == 0:
        raise ValueError(f"no available market date on or before {query}")
    return eligible[-1], True


def changed_from_previous(weights: pd.DataFrame, dt: pd.Timestamp) -> bool:
    loc = weights.index.get_loc(dt)
    if loc == 0:
        return True
    return bool(float((weights.iloc[loc] - weights.iloc[loc - 1]).abs().max()) > 1e-9)


def top_allocation(row: pd.Series) -> str:
    ordered = row[row > 0.001].sort_values(ascending=False)
    return ", ".join(f"{asset} {weight:.1%}" for asset, weight in ordered.items())


def build_weights(model: str, prices: pd.DataFrame, overlay: Optional[NarrativeOverlay]) -> pd.DataFrame:
    if model == "v9":
        return run_strategy(prices, BASE_V9, overlay=overlay)
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    if model == "v12":
        return run_meta_strategy(prices, macro, meta_candidates()[3], overlay=overlay, tx_cost=DEFAULT_TX)
    if model == "v13":
        return run_sparse_meta_strategy(prices, macro, risk_preserving_candidates()[0], overlay=overlay, tx_cost=DEFAULT_TX)
    if model == "v14":
        return run_sparse_sleeve_strategy(prices, macro, sleeve_sparse_candidates()[0], overlay=overlay, tx_cost=DEFAULT_TX)
    raise ValueError(f"unsupported model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show model weights for a given date.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--model", choices=["v9", "v12", "v13", "v14"], default="v9")
    parser.add_argument("--universe-mode", choices=["benchmark", "research", "tradable"], default="benchmark")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--llm-override-file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, refresh=False)
    overlay = load_llm_overlay(args.llm_override_file, prices.index)
    weights = build_weights(args.model, prices, overlay).reindex(prices.index).ffill().fillna(0.0)
    resolved_dt, adjusted = resolve_date(weights.index, args.date)
    row = weights.loc[resolved_dt]

    payload: Dict[str, object] = {
        "model": args.model,
        "query_date": args.date,
        "resolved_date": resolved_dt.strftime("%Y-%m-%d"),
        "adjusted_to_previous_trading_day": adjusted,
        "queried_day_is_trading_day": is_trading_day(args.date),
        "holiday_name": holiday_name(args.date),
        "changed_from_previous_trading_day": changed_from_previous(weights, resolved_dt),
        "next_trading_day": next_trading_day(resolved_dt).isoformat(),
        "top_allocation": top_allocation(row),
        "weights": {asset: float(row[asset]) for asset in weights.columns},
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Model: {payload['model']}")
    print(f"Query date: {payload['query_date']}")
    print(f"Resolved date: {payload['resolved_date']}")
    if adjusted:
        print("Adjusted to previous trading day because the queried date had no market bar.")
    if payload["holiday_name"]:
        print(f"Holiday: {payload['holiday_name']}")
    print(f"Changed from previous trading day: {'Yes' if payload['changed_from_previous_trading_day'] else 'No'}")
    print(f"Next trading day: {payload['next_trading_day']}")
    print(f"Allocation: {payload['top_allocation']}")
    print("\nWeights:")
    for asset, weight in payload["weights"].items():
        print(f"  {asset:<10} {weight:.4%}")


if __name__ == "__main__":
    main()
