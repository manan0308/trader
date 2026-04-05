#!/usr/bin/env python3
"""
US Portability Check For v9
===========================

Run the current v9 framework on a long-history US proxy universe.

Important caveat:
- This is a portability test, not a live deployment universe.
- We map the Indian-style sleeves onto US proxies so we can judge whether the
  logic generalizes outside one market regime.

Proxy mapping:
- NIFTY    -> SPY
- MIDCAP   -> MDY
- SMALLCAP -> IWM
- GOLD     -> GLD
- SILVER   -> SLV
- US       -> QQQ
- CASH     -> SHY

Coverage starts in 2006 because SLV begins in April 2006.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_data.market_store import load_processed_matrix
from strategy.v9_engine import (
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


RESULTS_PATH = CACHE_DIR / "us_portability_v9_results.json"
US_PROXY_TICKERS: Dict[str, str] = {
    "NIFTY": "SPY",
    "MIDCAP": "MDY",
    "SMALLCAP": "IWM",
    "GOLD": "GLD",
    "SILVER": "SLV",
    "US": "QQQ",
    "CASH": "SHY",
}


def fetch_us_proxy_prices(start: str) -> pd.DataFrame:
    local = load_processed_matrix("us_analog_shy", start=start)
    if local is not None and len(local) > 0:
        return local[ALL]
    raw = {}
    for sleeve, ticker in US_PROXY_TICKERS.items():
        history = yf.Ticker(ticker).history(start=start, auto_adjust=True, repair=True)
        if history.empty or "Close" not in history.columns:
            raise RuntimeError(f"empty history for {ticker}")
        raw[sleeve] = clean_price_series(history["Close"])
    return pd.DataFrame(raw).ffill(limit=5).dropna()[ALL]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v9 logic on a US proxy universe.")
    parser.add_argument("--start", default="2006-05-01")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = fetch_us_proxy_prices(args.start)

    candidates = [
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

    v9_wfo = parametric_v9_wfo(prices, candidates, rf=args.rf, tx_cost=tx_cost, overlay=None, train_days=756, test_days=126)
    eqwt = performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost)
    spy = performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "SPY B&H", rf=args.rf, tx_cost=tx_cost)

    print(f"US proxy sample: {prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d} | {len(prices)} rows")
    print(
        f"v9 strict OOS: CAGR {v9_wfo['metrics']['cagr']:.1%} | "
        f"Sharpe {v9_wfo['metrics']['sharpe']:.2f} | "
        f"MaxDD {v9_wfo['metrics']['mdd']:.1%} | "
        f"Turnover {v9_wfo['metrics']['turnover']:.0%}"
    )
    print(
        f"EqWt Risky:     CAGR {eqwt['cagr']:.1%} | "
        f"Sharpe {eqwt['sharpe']:.2f} | "
        f"MaxDD {eqwt['mdd']:.1%}"
    )
    print(
        f"SPY B&H:        CAGR {spy['cagr']:.1%} | "
        f"Sharpe {spy['sharpe']:.2f} | "
        f"MaxDD {spy['mdd']:.1%}"
    )

    payload = {
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "v9_wfo": {
            key: float(v9_wfo["metrics"][key])  # type: ignore[index]
            for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]
        },
        "eqwt": {key: float(eqwt[key]) for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]},
        "spy": {key: float(spy[key]) for key in ["cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"]},
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
