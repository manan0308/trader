#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from strategy.v9_engine import ALL, YahooFinanceSource, load_llm_overlay, run_strategy
from research.alpha_v12_meta_ensemble import BASE_V9
from execution.rebalance_core import ExecutionConfig, plan_rebalance
from runtime.india_market_calendar import holiday_name, is_trading_day
from runtime.paper_ledger import (
    PAPER_JOURNAL_PATH,
    PAPER_LATEST_PATH,
    default_paper_state,
    fill_pending_orders,
    load_paper_state,
    queue_pending_orders,
    save_paper_state,
    summarize_paper_state,
)
from runtime.store import PAPER_HISTORY_PATH, PAPER_STATE_PATH, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
BACKFILL_PATH = CACHE_DIR / "paper_backfill_latest.json"


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def top_allocation(weights: Dict[str, float]) -> str:
    ordered = sorted(((asset, value) for asset, value in weights.items() if value > 0.001), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{asset} {value:.1%}" for asset, value in ordered[:5])


def order_payload(plan) -> List[Dict[str, Any]]:
    return [
        {
            "asset": order.asset,
            "side": order.side,
            "quantity": order.quantity,
            "reference_price": order.reference_price,
            "order_price": order.order_price,
            "delta_weight": float(plan.target_weights[order.asset] - plan.starting_weights[order.asset]),
            "delta_value": order.delta_value,
        }
        for order in plan.orders
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill paper trading from a start date using v9.")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-04-03")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--llm-override-file")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)

    benchmark_prices = YahooFinanceSource(universe_mode="benchmark").fetch("2012-01-01", refresh=False)
    tradable_prices = YahooFinanceSource(universe_mode="tradable").fetch("2024-01-01", refresh=False)
    overlay = load_llm_overlay(args.llm_override_file, benchmark_prices.index)
    weights = run_strategy(benchmark_prices, BASE_V9, overlay=overlay).reindex(benchmark_prices.index).ffill().fillna(0.0)

    trading_index = benchmark_prices.index.intersection(tradable_prices.index)
    trading_dates = {dt.date(): dt for dt in trading_index if start_date <= dt.date() <= end_date}

    if args.reset and PAPER_STATE_PATH.exists():
        PAPER_STATE_PATH.unlink(missing_ok=True)
        PAPER_HISTORY_PATH.unlink(missing_ok=True)
        PAPER_JOURNAL_PATH.unlink(missing_ok=True)
        PAPER_LATEST_PATH.unlink(missing_ok=True)

    state = default_paper_state(starting_cash=args.initial_cash)
    trading_history: List[Dict[str, Any]] = []
    calendar_curve: List[Dict[str, Any]] = []
    prev_target: pd.Series | None = None

    for cal_day in daterange(start_date, end_date):
        if cal_day in trading_dates and is_trading_day(cal_day):
            dt = trading_dates[cal_day]
            price_row = tradable_prices.loc[dt]
            price_map = {asset: float(price_row[asset]) for asset in ALL}
            fills = fill_pending_orders(state, price_map=price_map, as_of=cal_day.isoformat())

            target_row = weights.loc[dt]
            plan = plan_rebalance(
                target_weights={asset: float(target_row[asset]) for asset in ALL},
                holdings=state["positions"],
                prices=price_map,
                available_cash=float(state["cash"]),
                config=ExecutionConfig(),
                as_of=dt,
            )
            orders = order_payload(plan)
            queue_pending_orders(state, orders, created_at=cal_day.isoformat())
            summary = summarize_paper_state(state, price_map=price_map, recent_fills=fills)

            changed = True if prev_target is None else bool(float((target_row - prev_target).abs().max()) > 1e-9)
            prev_target = target_row

            trading_row = {
                "as_of": cal_day.isoformat(),
                "trading_day": True,
                "holiday_name": None,
                "equity": float(summary["total_equity"]),
                "cash": float(summary["cash"]),
                "weights": {asset: float(target_row[asset]) for asset in ALL},
                "top_allocation": top_allocation({asset: float(target_row[asset]) for asset in ALL}),
                "changed": changed,
                "filled_orders": fills,
                "queued_orders": orders,
                "pending_orders": list(state.get("pending_orders", [])),
            }
            trading_history.append(trading_row)
            calendar_curve.append(
                {
                    "as_of": cal_day.isoformat(),
                    "trading_day": True,
                    "holiday_name": None,
                    "equity": float(summary["total_equity"]),
                    "cash": float(summary["cash"]),
                    "pending_orders": len(state.get("pending_orders", [])),
                }
            )
        else:
            carry_equity = float(calendar_curve[-1]["equity"]) if calendar_curve else float(args.initial_cash)
            carry_cash = float(state.get("cash", args.initial_cash))
            calendar_curve.append(
                {
                    "as_of": cal_day.isoformat(),
                    "trading_day": False,
                    "holiday_name": holiday_name(cal_day) or ("Weekend" if cal_day.weekday() >= 5 else None),
                    "equity": carry_equity,
                    "cash": carry_cash,
                    "pending_orders": len(state.get("pending_orders", [])),
                }
            )

    latest_summary = trading_history[-1] if trading_history else {
        "as_of": end_date.isoformat(),
        "equity": float(args.initial_cash),
        "cash": float(args.initial_cash),
    }
    final_equity = float(latest_summary["equity"])
    payload = {
        "model": "v9",
        "start": args.start,
        "end": args.end,
        "initial_cash": float(args.initial_cash),
        "final_equity": final_equity,
        "net_pnl": final_equity - float(args.initial_cash),
        "return_since_start": final_equity / float(args.initial_cash) - 1.0,
        "trading_history": trading_history,
        "calendar_curve": calendar_curve,
        "final_pending_orders": list(state.get("pending_orders", [])),
        "final_positions": dict(state.get("positions", {})),
        "final_cash": float(state.get("cash", 0.0)),
    }
    write_json(BACKFILL_PATH, payload)

    if args.apply:
        save_paper_state(state)
        write_jsonl(
            PAPER_HISTORY_PATH,
            [
                {
                    "as_of": row["as_of"],
                    "equity": row["equity"],
                    "cash": row["cash"],
                    "weights": row["weights"],
                    "filled_count": len(row["filled_orders"]),
                    "queued_count": len(row["queued_orders"]),
                }
                for row in trading_history
            ],
        )
        latest_payload = {
            "as_of": latest_summary["as_of"],
            "cash": float(state.get("cash", 0.0)),
            "equity": final_equity,
            "total_equity": final_equity,
            "net_pnl": final_equity - float(args.initial_cash),
            "return_since_start": final_equity / float(args.initial_cash) - 1.0,
            "positions": [
                {
                    "asset": asset,
                    "quantity": int(state["positions"].get(asset, 0)),
                    "price": float(tradable_prices.loc[trading_dates[end_date] if end_date in trading_dates else list(trading_dates.values())[-1], asset]) if trading_dates else 0.0,
                    "market_value": int(state["positions"].get(asset, 0))
                    * float(tradable_prices.loc[trading_dates[end_date] if end_date in trading_dates else list(trading_dates.values())[-1], asset])
                    if trading_dates
                    else 0.0,
                }
                for asset in ALL
            ],
            "pending_orders": list(state.get("pending_orders", [])),
            "recent_fills": trading_history[-1]["filled_orders"] if trading_history else [],
            "history": [
                {
                    "as_of": row["as_of"],
                    "equity": row["equity"],
                    "cash": row["cash"],
                    "filled_count": len(row["filled_orders"]),
                    "queued_count": len(row["queued_orders"]),
                }
                for row in trading_history
            ],
            "equity_curve": calendar_curve,
        }
        write_json(PAPER_LATEST_PATH, latest_payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
