#!/usr/bin/env python3
"""
Live Signal Report
==================

Generate recent live-style allocation signals and the realized next-day outcome
for the latest completed trading sessions.

Supported models:
- v9
- v12
- all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.v9_engine import (
    ALL,
    DEFAULT_TX,
    GrowwSource,
    NarrativeOverlay,
    StrategyConfig,
    YahooFinanceSource,
    load_llm_overlay,
    portfolio_returns,
    run_strategy,
)
from broker.groww_client import GrowwSession, load_production_groww_universe
from research.alpha_v11_macro_value_research import fetch_macro_panel
from research.alpha_v12_meta_ensemble import BASE_V9, meta_candidates, run_meta_strategy
from research.alpha_v13_sparse_meta import risk_preserving_candidates, run_sparse_meta_strategy
from research.alpha_v14_sparse_sleeves import run_sparse_sleeve_strategy, sleeve_sparse_candidates
from runtime.india_market_calendar import market_clock, next_trading_day
from runtime.display_labels import asset_label
from runtime.env_loader import load_runtime_env


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
LATEST_SIGNAL_PATH = CACHE_DIR / "live_signal_latest.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    builder: Callable[[pd.DataFrame, Optional[NarrativeOverlay]], pd.DataFrame]


def v9_builder(prices: pd.DataFrame, overlay: Optional[NarrativeOverlay]) -> pd.DataFrame:
    return run_strategy(prices, BASE_V9, overlay=overlay)


def v12_builder(prices: pd.DataFrame, overlay: Optional[NarrativeOverlay]) -> pd.DataFrame:
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    config = meta_candidates()[3]
    return run_meta_strategy(prices, macro, config, overlay=overlay, tx_cost=DEFAULT_TX)


def v13_builder(prices: pd.DataFrame, overlay: Optional[NarrativeOverlay]) -> pd.DataFrame:
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    config = risk_preserving_candidates()[0]
    return run_sparse_meta_strategy(prices, macro, config, overlay=overlay, tx_cost=DEFAULT_TX)


def v14_builder(prices: pd.DataFrame, overlay: Optional[NarrativeOverlay]) -> pd.DataFrame:
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    config = sleeve_sparse_candidates()[0]
    return run_sparse_sleeve_strategy(prices, macro, config, overlay=overlay, tx_cost=DEFAULT_TX)


def model_specs() -> Dict[str, ModelSpec]:
    return {
        "v9": ModelSpec(name="v9", label="v9 weekly_core85_tilt15", builder=v9_builder),
        "v12": ModelSpec(name="v12", label="v12 meta_63126_smooth", builder=v12_builder),
        "v13": ModelSpec(name="v13", label="v13 sparse_top4_monthly_slow", builder=v13_builder),
        "v14": ModelSpec(name="v14", label="v14 sleeve_top1_slow", builder=v14_builder),
    }


def resolve_data_source(requested: str, groww_universe_file: str | None = None) -> str:
    if requested in {"yfinance", "groww"}:
        return requested
    load_runtime_env(override=True)
    try:
        instruments = load_production_groww_universe(groww_universe_file)
        GrowwSession.from_env(instrument_map=instruments, cache_dir=CACHE_DIR)
        return "groww"
    except Exception:
        return "yfinance"


def fetch_prices(
    *,
    start: str,
    refresh: bool,
    data_source: str,
    universe_mode: str,
    groww_universe_file: str | None = None,
) -> pd.DataFrame:
    if data_source == "groww":
        try:
            instruments = load_production_groww_universe(groww_universe_file)
            session = GrowwSession.from_env(instrument_map=instruments, cache_dir=CACHE_DIR)
            prices = GrowwSource(session).fetch(start=start, refresh=refresh)
            prices.attrs["data_source_actual"] = "groww"
            return prices
        except Exception as exc:
            prices = YahooFinanceSource(universe_mode=universe_mode).fetch(start, refresh=refresh)
            prices.attrs["data_source_actual"] = "yfinance"
            prices.attrs["data_source_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return prices
    prices = YahooFinanceSource(universe_mode=universe_mode).fetch(start, refresh=refresh)
    prices.attrs["data_source_actual"] = "yfinance"
    return prices


def top_weights(row: pd.Series, top_n: int = 3) -> str:
    ordered = row[row > 0.001].sort_values(ascending=False).head(top_n)
    if ordered.empty:
        return f"{asset_label('CASH')} 100.0%"
    return ", ".join(f"{asset_label(asset)} {weight:.1%}" for asset, weight in ordered.items())


def format_delta(prev_row: pd.Series, row: pd.Series, threshold: float = 0.01) -> str:
    delta = (row - prev_row).sort_values(key=lambda s: s.abs(), ascending=False)
    moved = delta[delta.abs() >= threshold].head(3)
    if moved.empty:
        return "No material change"
    parts = []
    for asset, change in moved.items():
        sign = "+" if change >= 0 else ""
        parts.append(f"{asset_label(asset)} {sign}{change:.1%}")
    return ", ".join(parts)


def recent_signal_rows(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    days: int,
    tx_cost: float,
) -> List[Dict[str, object]]:
    daily_returns = portfolio_returns(prices.loc[weights.index], weights, tx_cost=tx_cost)
    idx = weights.index
    recent_dates = list(idx[-days:])
    rows: List[Dict[str, object]] = []

    for dt in recent_dates:
        loc = idx.get_loc(dt)
        row = weights.loc[dt]
        prev_row = weights.iloc[loc - 1] if loc > 0 else pd.Series(0.0, index=weights.columns)
        changed = bool(float((row - prev_row).abs().max()) > 1e-6)

        next_date = idx[loc + 1] if loc + 1 < len(idx) else pd.Timestamp(next_trading_day(dt))
        realized = float(daily_returns.loc[next_date]) if next_date is not None and next_date in daily_returns.index else np.nan

        rows.append(
            {
                "signal_date": dt,
                "next_date": next_date,
                "changed": changed,
                "allocation": top_weights(row),
                "delta": format_delta(prev_row, row),
                "next_day_return": realized,
            }
        )
    return rows


def print_report(spec: ModelSpec, prices: pd.DataFrame, weights: pd.DataFrame, days: int, tx_cost: float) -> None:
    rows = recent_signal_rows(prices, weights, days=days, tx_cost=tx_cost)
    latest = rows[-1]
    latest_weights = weights.iloc[-1]
    latest_date = latest["signal_date"]
    next_session = latest["next_date"]

    print("\n" + "=" * 104)
    print(spec.label)
    print("=" * 104)
    print(f"Latest completed bar: {latest_date:%Y-%m-%d}")
    if next_session is not None:
        print(f"Actionable for next session: {next_session:%Y-%m-%d}")
    else:
        print("Actionable for next session: pending next trading day")
    print(f"Current allocation: {top_weights(latest_weights, top_n=5)}")

    print("\nRecent live-style signals:")
    print(f"{'Signal Date':>12} {'Action Date':>12} {'Changed':>8} {'Next Day':>10} {'Allocation':>38} {'Delta':>40}")
    print("-" * 132)
    for row in rows:
        next_day = "pending" if np.isnan(row["next_day_return"]) else f"{row['next_day_return']:.2%}"
        action_date = row["next_date"].strftime("%Y-%m-%d") if row["next_date"] is not None else "pending"
        changed = "Yes" if row["changed"] else "No"
        print(
            f"{row['signal_date']:%Y-%m-%d} {action_date:>12} {changed:>8} {next_day:>10} "
            f"{row['allocation']:>38} {row['delta']:>40}"
        )

    return {
        "model": spec.name,
        "label": spec.label,
        "latest_completed_bar": latest_date.strftime("%Y-%m-%d"),
        "actionable_for_next_session": next_session.strftime("%Y-%m-%d") if next_session is not None else None,
        "current_allocation": top_weights(latest_weights, top_n=5),
        "rows": [
            {
                "signal_date": row["signal_date"].strftime("%Y-%m-%d"),
                "action_date": row["next_date"].strftime("%Y-%m-%d") if row["next_date"] is not None else None,
                "changed": bool(row["changed"]),
                "allocation": row["allocation"],
                "delta": row["delta"],
                "next_day_return": None if np.isnan(row["next_day_return"]) else float(row["next_day_return"]),
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show latest live-style model signals and realized next-day outcomes.")
    parser.add_argument("--model", choices=["v9", "v12", "v13", "v14", "all"], default="all")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--data-source", choices=["auto", "yfinance", "groww"], default="auto")
    parser.add_argument("--universe-mode", choices=["research", "benchmark", "tradable"], default="benchmark")
    parser.add_argument("--groww-universe-file")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--llm-override-file")
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--no-cache-write", action="store_true")
    args = parser.parse_args()

    data_source = resolve_data_source(args.data_source, args.groww_universe_file)
    prices = fetch_prices(
        start=args.start,
        refresh=args.refresh_cache,
        data_source=data_source,
        universe_mode=args.universe_mode,
        groww_universe_file=args.groww_universe_file,
    )
    actual_data_source = prices.attrs.get("data_source_actual", data_source)
    fallback_reason = prices.attrs.get("data_source_fallback_reason")
    overlay = load_llm_overlay(args.llm_override_file, prices.index)
    tx_cost = args.tx_bps / 10_000
    clock = market_clock()

    dashboard_payload = {
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "data_source": actual_data_source,
        "requested_data_source": data_source,
        "data_source_fallback_reason": fallback_reason,
        "tx_cost": tx_cost,
        "universe_mode": args.universe_mode,
        "market_clock": clock.__dict__,
        "models": {},
    }
    specs = model_specs()
    selected = specs.values() if args.model == "all" else [specs[args.model]]

    print(f"Latest available market data is through {prices.index[-1]:%Y-%m-%d} ({actual_data_source}).")
    if clock.session in {"holiday", "weekend"}:
        reason = f" ({clock.holiday_name})" if clock.holiday_name else ""
        print(f"India market status: {clock.session}{reason}. Next trading day is {clock.next_trading_day}.")
    else:
        print(f"India market status: {clock.session}. Next trading day is {clock.next_trading_day}.")
    if prices.index[-1].date().isoformat() != pd.Timestamp.now(tz=None).date().isoformat():
        print("Note: if today's close is not yet published, the latest signal is as of the most recent completed trading day.")

    for spec in selected:
        weights = spec.builder(prices, overlay)
        weights = weights.reindex(prices.index).ffill().fillna(0.0)
        dashboard_payload["models"][spec.name] = print_report(spec, prices, weights, days=args.days, tx_cost=tx_cost)

    if not args.no_cache_write:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_SIGNAL_PATH.write_text(json.dumps(dashboard_payload, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved dashboard snapshot to {LATEST_SIGNAL_PATH}")


if __name__ == "__main__":
    main()
