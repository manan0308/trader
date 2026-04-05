#!/usr/bin/env python3
"""
Performance Significance Diagnostics
===================================

Compute simple statistical significance diagnostics for the main stitched-OOS
strategies on the shared India dataset.

Why this script exists:
- Backtest Sharpe alone is not enough.
- Daily returns can have autocorrelation, so we report both:
  1. naive IID Sharpe t-statistics
  2. HAC/Newey-West t-tests on mean excess return

This does not "prove" a strategy is real, but it is a much better filter than
looking at CAGR / Sharpe alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from trader_system.strategy.v9_engine import (
    CACHE_DIR,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    DEFAULT_TX,
    YahooFinanceSource,
    benchmark_weights,
    performance_metrics,
)
from research.alpha_v11_macro_value_research import parametric_v9_wfo
from research.alpha_v12_meta_ensemble import BASE_V9


RESULTS_PATH = CACHE_DIR / "performance_significance.json"


def rf_daily(rf: float) -> float:
    return (1.0 + rf) ** (1.0 / 252.0) - 1.0


def hac_lags(length: int) -> int:
    return max(int(np.floor(4.0 * (length / 100.0) ** (2.0 / 9.0))), 1)


def iid_sharpe_test(returns: pd.Series, rf: float) -> Dict[str, float]:
    ex = pd.Series(returns).dropna() - rf_daily(rf)
    sr_daily = float(ex.mean() / ex.std()) if float(ex.std()) > 0 else np.nan
    t_stat = sr_daily * np.sqrt(len(ex)) if np.isfinite(sr_daily) else np.nan
    df = max(len(ex) - 1, 1)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan
    return {
        "sharpe": sr_daily * np.sqrt(252.0),
        "iid_t_stat": float(t_stat),
        "iid_p_value": p_value,
        "sample_size": float(len(ex)),
    }


def hac_mean_excess_test(returns: pd.Series, rf: float) -> Dict[str, float]:
    ex = pd.Series(returns).dropna() - rf_daily(rf)
    lags = hac_lags(len(ex))
    fit = sm.OLS(ex.values, np.ones(len(ex))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean_daily = float(fit.params[0])
    se = float(fit.bse[0])
    t_stat = mean_daily / se if se > 0 else np.nan
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else np.nan
    return {
        "annualized_excess": mean_daily * 252.0,
        "hac_t_stat": float(t_stat),
        "hac_p_value": p_value,
        "hac_lags": float(lags),
    }


def hac_difference_test(lhs: pd.Series, rhs: pd.Series) -> Dict[str, float]:
    lhs_aligned, rhs_aligned = pd.Series(lhs).dropna().align(pd.Series(rhs).dropna(), join="inner")
    diff = lhs_aligned - rhs_aligned
    lags = hac_lags(len(diff))
    fit = sm.OLS(diff.values, np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean_daily = float(fit.params[0])
    se = float(fit.bse[0])
    t_stat = mean_daily / se if se > 0 else np.nan
    p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat)))) if np.isfinite(t_stat) else np.nan
    return {
        "annualized_return_diff": mean_daily * 252.0,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "hac_lags": float(lags),
    }


def build_model_returns(prices: pd.DataFrame, rf: float, tx_cost: float) -> Tuple[pd.DatetimeIndex, Dict[str, pd.Series]]:
    v9 = parametric_v9_wfo(prices, [BASE_V9], rf=rf, tx_cost=tx_cost, overlay=None, train_days=756, test_days=126)

    common_dates = v9["metrics"]["returns"].index  # type: ignore[index]
    eqwt = performance_metrics(prices.loc[common_dates], benchmark_weights(prices.loc[common_dates], "EqWt Risky"), "EqWt Risky", rf=rf, tx_cost=tx_cost)
    nifty = performance_metrics(prices.loc[common_dates], benchmark_weights(prices.loc[common_dates], "Nifty B&H"), "Nifty B&H", rf=rf, tx_cost=tx_cost)

    return common_dates, {
        "v9": v9["metrics"]["returns"],  # type: ignore[index]
        "eqwt": eqwt["returns"],  # type: ignore[index]
        "nifty": nifty["returns"],  # type: ignore[index]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run significance diagnostics on the stitched OOS strategies.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--universe-mode", choices=["research", "benchmark", "tradable"], default="benchmark")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, end=args.end, refresh=args.refresh_cache)
    _, model_returns = build_model_returns(prices, rf=args.rf, tx_cost=tx_cost)

    summary: Dict[str, Dict[str, float]] = {}
    for name, series in model_returns.items():
        summary[name] = {
            **iid_sharpe_test(series, rf=args.rf),
            **hac_mean_excess_test(series, rf=args.rf),
        }

    pairwise = {
        "v9_minus_eqwt": hac_difference_test(model_returns["v9"], model_returns["eqwt"]),
        "v9_minus_nifty": hac_difference_test(model_returns["v9"], model_returns["nifty"]),
    }

    print("\nSignificance diagnostics:")
    for name, stats_row in summary.items():
        print(
            f"  {name:<6} "
            f"Sharpe {stats_row['sharpe']:.3f} | "
            f"IID t {stats_row['iid_t_stat']:.3f} p {stats_row['iid_p_value']:.4f} | "
            f"HAC ann excess {stats_row['annualized_excess']:.2%} "
            f"t {stats_row['hac_t_stat']:.3f} p {stats_row['hac_p_value']:.4f}"
        )

    print("\nPairwise HAC return-difference tests:")
    for name, stats_row in pairwise.items():
        print(
            f"  {name:<14} "
            f"ann diff {stats_row['annualized_return_diff']:.2%} | "
            f"t {stats_row['t_stat']:.3f} p {stats_row['p_value']:.4f}"
        )

    payload = {"models": summary, "pairwise": pairwise}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
