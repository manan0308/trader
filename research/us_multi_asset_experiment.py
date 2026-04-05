#!/usr/bin/env python3
"""
US Multi-Asset Portability Experiment
=====================================

Run the current v9 framework on a few US proxy universes to see whether the
multi-asset logic generalizes outside India.

Default universe mapping:
- NIFTY    -> SPY
- MIDCAP   -> MDY
- SMALLCAP -> IWM
- GOLD     -> GLD
- SILVER   -> SLV
- US       -> QQQ
- CASH     -> a chosen defensive USD sleeve

This is still a portability experiment, not a live production universe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader_system.data.market_store import load_processed_matrix
from trader_system.strategy.v9_engine import (
    ALL,
    CACHE_DIR,
    DEFAULT_RF,
    DEFAULT_TX,
    StrategyConfig,
    benchmark_weights,
    clean_price_series,
    performance_metrics,
)
from research.alpha_v11_macro_value_research import parametric_v9_wfo
from research.alpha_v12_meta_ensemble import BASE_V9


RESULTS_PATH = CACHE_DIR / "us_multi_asset_experiment.json"

UNIVERSES: Dict[str, Dict[str, str]] = {
    "us_analog_shy": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "SHY",
    },
    "us_analog_ief": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "IEF",
    },
    "us_analog_bil": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "BIL",
    },
}


def candidate_configs() -> List[StrategyConfig]:
    return [
        BASE_V9,
        StrategyConfig(
            name="weekly_core80_tilt20",
            execution_frequency="WEEKLY",
            core_weight=0.80,
            tilt_weight=0.20,
            top_n=2,
            trade_band=0.08,
            trade_step=0.75,
            crash_floor=0.65,
        ),
        StrategyConfig(
            name="monthly_core70_tilt30",
            execution_frequency="MONTHLY",
            core_weight=0.70,
            tilt_weight=0.30,
            top_n=2,
            trade_band=0.06,
            trade_step=1.00,
            crash_floor=0.65,
        ),
        StrategyConfig(
            name="monthly_core85_tilt15",
            execution_frequency="MONTHLY",
            core_weight=0.85,
            tilt_weight=0.15,
            top_n=2,
            trade_band=0.05,
            trade_step=1.00,
            crash_floor=0.70,
        ),
    ]


def fetch_us_proxy_prices(mapping: Dict[str, str], start: str) -> pd.DataFrame:
    for dataset_name, universe in UNIVERSES.items():
        if universe == mapping:
            local = load_processed_matrix(dataset_name, start=start)
            if local is not None and len(local) > 0:
                return local[ALL]
    raw = {}
    for sleeve, ticker in mapping.items():
        history = yf.Ticker(ticker).history(start=start, auto_adjust=True, repair=True)
        if history.empty or "Close" not in history.columns:
            raise RuntimeError(f"empty history for {ticker}")
        raw[sleeve] = clean_price_series(history["Close"])
    return pd.DataFrame(raw).ffill(limit=5).dropna()[ALL]


def summarize_result(label: str, metrics: Dict[str, float]) -> str:
    return (
        f"{label}: CAGR {metrics['cagr']:.1%} | "
        f"Sharpe {metrics['sharpe']:.2f} | "
        f"MaxDD {metrics['mdd']:.1%} | "
        f"Turnover {metrics['turnover']:.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v9 on US-equivalent multi-asset universes.")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    configs = candidate_configs()
    payload: Dict[str, object] = {}

    print("=" * 88)
    print("US MULTI-ASSET PORTABILITY EXPERIMENT")
    print("=" * 88)
    print(f"Requested start date: {args.start}")
    print(f"Transaction cost: {tx_cost:.2%} per trade")
    print("Note: there is no long-history USD cash ETF that stays near 5-6% through the sample.")
    print("      SHY/IEF/BIL represent defensive sleeves under different rate regimes.\n")

    for universe_name, mapping in UNIVERSES.items():
        prices = fetch_us_proxy_prices(mapping, start=args.start)
        wfo = parametric_v9_wfo(prices, configs, rf=args.rf, tx_cost=tx_cost, overlay=None, train_days=756, test_days=126)
        eqwt = performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost)
        largecap = performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), mapping["NIFTY"], rf=args.rf, tx_cost=tx_cost)

        print(f"{universe_name} | sample {prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d} | rows {len(prices)}")
        print("  " + summarize_result("v9 strict OOS", wfo["metrics"]))  # type: ignore[arg-type]
        print("  " + summarize_result("EqWt Risky", eqwt))
        print("  " + summarize_result(f"{mapping['NIFTY']} B&H", largecap))
        print("")

        payload[universe_name] = {
            "mapping": mapping,
            "sample_start": prices.index[0].strftime("%Y-%m-%d"),
            "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
            "rows": len(prices),
            "v9_wfo": {
                key: float(wfo["metrics"][key])  # type: ignore[index]
                for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]
            },
            "eqwt": {
                key: float(eqwt[key])
                for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]
            },
            "largecap_bh": {
                key: float(largecap[key])
                for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]
            },
            "window_tail": [
                {
                    "window": int(row["window"]),
                    "picked": row["picked"],
                    "test_start": str(row["test_start"]),
                    "test_end": str(row["test_end"]),
                    "cagr": float(row["cagr"]),
                    "sharpe": float(row["sharpe"]),
                    "mdd": float(row["mdd"]),
                }
                for row in wfo["windows"][-5:]  # type: ignore[index]
            ],
        }

    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
