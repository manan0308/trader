#!/usr/bin/env python3
"""
Execution Planner CLI
=====================

Thin wrapper around the reusable execution core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from strategy.v9_engine import CACHE_DIR, GrowwSource, YahooFinanceSource, load_llm_overlay, run_strategy, StrategyConfig
from runtime.audit_log import stable_hash
from execution.rebalance_core import ExecutionConfig, format_plan, plan_rebalance
from execution.groww_adapter import (
    build_groww_order_requests,
    current_holdings_snapshot,
    extract_asset_quantities,
    order_request_to_payload,
)
from broker.groww_client import (
    PRODUCTION_GROWW_UNIVERSE_PATH,
    GrowwInstrument,
    GrowwSession,
    load_production_groww_universe,
)


PLAN_PATH = CACHE_DIR / "execution_plan_latest.json"


def recommended_config() -> StrategyConfig:
    return StrategyConfig(
        name="weekly_core85_tilt15",
        execution_frequency="WEEKLY",
        core_weight=0.85,
        tilt_weight=0.15,
        top_n=2,
        trade_band=0.08,
        trade_step=0.75,
        crash_floor=0.70,
    )


def latest_target_weights(prices, overlay):
    weights = run_strategy(prices, recommended_config(), overlay=overlay)
    return weights.iloc[-1].to_dict()


def build_price_map(prices, instruments: Dict[str, GrowwInstrument], groww_session: Optional[GrowwSession], use_live_ltp: bool) -> Dict[str, float]:
    latest = prices.iloc[-1].to_dict()
    if not groww_session or not use_live_ltp:
        return {asset: float(latest[asset]) for asset in latest}

    try:
        ltp = groww_session.multi_ltp(instruments.keys())
    except Exception:
        return {asset: float(latest[asset]) for asset in latest}

    out: Dict[str, float] = {}
    for asset, instrument in instruments.items():
        candidates = [
            f"{instrument.exchange}_{instrument.trading_symbol}",
            instrument.groww_symbol,
            instrument.trading_symbol,
        ]
        price = None
        for key in candidates:
            if key in ltp:
                try:
                    price = float(ltp[key])
                    break
                except Exception:
                    continue
        out[asset] = float(price) if price is not None else float(latest[asset])
    return out


def load_portfolio_file(path: str) -> tuple[float, Dict[str, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cash = float(payload.get("cash", 0.0))
    holdings = {
        str(row["asset"]).upper(): int(float(row.get("quantity", 0)))
        for row in payload.get("positions", [])
    }
    return cash, holdings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dry-run execution plan from v9 target weights.")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--data-source", choices=["yfinance", "groww"], default="yfinance")
    parser.add_argument("--universe-mode", choices=["benchmark", "research", "tradable"], default="tradable")
    parser.add_argument("--groww-universe-file")
    parser.add_argument("--portfolio-file")
    parser.add_argument("--groww-live", action="store_true")
    parser.add_argument("--cash", type=float, default=0.0)
    parser.add_argument("--llm-override-file")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    instruments = load_production_groww_universe(args.groww_universe_file)
    groww_session = None
    if args.data_source == "groww" or args.groww_live:
        groww_session = GrowwSession.from_env(instrument_map=instruments, cache_dir=CACHE_DIR)

    data_source_used = args.data_source
    data_source_fallback_reason = None
    if args.data_source == "groww":
        try:
            prices = GrowwSource(groww_session).fetch(args.start, refresh=args.refresh_cache)  # type: ignore[arg-type]
        except Exception as exc:
            prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, refresh=args.refresh_cache)
            data_source_used = "yfinance"
            data_source_fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, refresh=args.refresh_cache)

    overlay = load_llm_overlay(args.llm_override_file, prices.index)
    target_weights = latest_target_weights(prices, overlay)
    price_map = build_price_map(prices, instruments, groww_session=groww_session, use_live_ltp=args.groww_live)

    if args.portfolio_file:
        cash, holdings = load_portfolio_file(args.portfolio_file)
    elif args.groww_live:
        snapshot = current_holdings_snapshot(groww_session, instrument_map=instruments)  # type: ignore[arg-type]
        holdings = extract_asset_quantities(snapshot["holdings_payload"], snapshot["positions_payload"], instrument_map=instruments)
        cash = float(args.cash or 0.0)
    else:
        holdings = {}
        cash = float(args.cash or 0.0)

    plan = plan_rebalance(
        target_weights=target_weights,
        holdings=holdings,
        prices=price_map,
        available_cash=cash,
        config=ExecutionConfig(),
        instrument_map=instruments,
    )
    groww_requests = build_groww_order_requests(plan.orders, instrument_map=instruments)
    groww_payloads = [order_request_to_payload(request) for request in groww_requests]

    print(format_plan(plan))

    plan_key = stable_hash(
        {
            "as_of": prices.index[-1].strftime("%Y-%m-%d"),
            "target_weights": target_weights,
            "orders": groww_payloads,
            "broker_map": {asset: instrument.trading_symbol for asset, instrument in instruments.items()},
        }
    )[:16]

    payload = {
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "plan_key": plan_key,
        "data_source": data_source_used,
        "requested_data_source": args.data_source,
        "data_source_fallback_reason": data_source_fallback_reason,
        "universe_mode": args.universe_mode,
        "groww_universe_file": args.groww_universe_file or str(PRODUCTION_GROWW_UNIVERSE_PATH),
        "broker_map": {
            asset: {
                "exchange": instrument.exchange,
                "segment": instrument.segment,
                "trading_symbol": instrument.trading_symbol,
                "groww_symbol": instrument.groww_symbol,
            }
            for asset, instrument in instruments.items()
        },
        "plan": {
            "portfolio_value": plan.total_equity,
            "current_weights": plan.starting_weights.to_dict(),
            "target_weights": plan.target_weights.to_dict(),
            "target_quantities": {asset: int(plan.target_quantities[asset]) for asset in plan.target_quantities.index},
            "post_trade_quantities": {
                asset: int(plan.post_trade_quantities[asset]) for asset in plan.post_trade_quantities.index
            },
            "reserve_cash": plan.reserve_cash,
            "post_trade_cash": plan.post_trade_cash,
            "orders": [
                {
                    "asset": order.asset,
                    "side": order.side,
                    "quantity": order.quantity,
                    "reference_price": order.reference_price,
                    "order_price": order.order_price,
                    "current_quantity": order.current_quantity,
                    "target_quantity": order.target_quantity,
                    "current_value": order.current_value,
                    "target_value": order.target_value,
                    "delta_value": order.delta_value,
                    "estimated_cost": order.estimated_cost,
                    "notes": order.notes,
                    "delta_weight": float(plan.target_weights[order.asset] - plan.starting_weights[order.asset]),
                }
                for order in plan.orders
            ],
            "groww_payloads": groww_payloads,
        },
    }
    output_path = Path(args.output).expanduser().resolve() if args.output else PLAN_PATH
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
