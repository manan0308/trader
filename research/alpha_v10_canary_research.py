#!/usr/bin/env python3
"""
ALPHA ENGINE v10 — CANARY + DUAL MOMENTUM RESEARCH
==================================================

This script extends the v9 research in two ways:

1. It adds a fixed-weight composite benchmark that includes all seven sleeves.
2. It tests a slower canary/dual-momentum allocator inspired by monthly TAA work.

Why this direction:
- Cross-asset time-series momentum has deep literature support.
- Simple monthly trend rules have historically improved risk-adjusted outcomes.
- Maximum-diversification, HMM, and ML-heavy layers are much harder to justify
  in a tiny long-only ETF universe with only daily bars and 30 bps trading costs.

Important implementation note:
- `US` is `MON100.NS`, already an INR-denominated ETF return stream.
  There is no separate FX uplift added on top of MON100 returns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trader_system.strategy.v9_engine import (
    ALL,
    RISKY,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    DEFAULT_TX,
    TEST_DAYS,
    TRAIN_DAYS,
    CACHE_DIR,
    RESULTS_CACHE,
    NarrativeOverlay,
    StrategyConfig,
    benchmark_oos_metrics,
    benchmark_weights,
    format_metrics,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
    strategy_score,
    walk_forward,
    YahooFinanceSource,
)


COMPOSITE_FIXED_WEIGHTS: Dict[str, float] = {
    "NIFTY": 0.25,
    "MIDCAP": 0.10,
    "SMALLCAP": 0.10,
    "GOLD": 0.15,
    "SILVER": 0.05,
    "US": 0.20,
    "CASH": 0.15,
}

CANARY_ASSETS = ["NIFTY", "MIDCAP", "SMALLCAP", "US"]
DEFENSIVE_ASSETS = ["GOLD", "US", "CASH"]
RESEARCH_RESULTS = CACHE_DIR / "alpha_v10_research_results.json"


@dataclass(frozen=True)
class CanaryConfig:
    name: str
    execution_frequency: str
    top_risk: int
    top_def: int
    trade_band: float
    score_mode: str
    canary_mode: str
    use_trend: bool


def fixed_weight_frame(prices: pd.DataFrame, alloc: Dict[str, float], label: str, rf: float, tx_cost: float) -> Dict[str, object]:
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    for asset, weight in alloc.items():
        weights[asset] = weight
    return performance_metrics(prices, weights, label, rf=rf, tx_cost=tx_cost)


def _score_series(
    mom21: pd.DataFrame,
    mom63: pd.DataFrame,
    mom126: pd.DataFrame,
    mom252: pd.DataFrame,
    dt: pd.Timestamp,
    mode: str,
) -> pd.Series:
    if mode == "midlong":
        score = 0.50 * mom126.loc[dt] + 0.50 * mom252.loc[dt]
    elif mode == "fastmid":
        score = 0.50 * mom63.loc[dt] + 0.50 * mom126.loc[dt]
    elif mode == "blend13612":
        score = 0.25 * mom21.loc[dt] + 0.25 * mom63.loc[dt] + 0.25 * mom126.loc[dt] + 0.25 * mom252.loc[dt]
    else:
        raise ValueError(f"unsupported score mode: {mode}")
    return score.replace([np.inf, -np.inf], np.nan).fillna(-999.0)


def _canary_defensive_share(score: pd.Series, mode: str) -> float:
    bad = int(sum(score[asset] <= 0 for asset in CANARY_ASSETS))
    if mode == "fractional":
        return bad / len(CANARY_ASSETS)
    if mode == "binary2":
        return 1.0 if bad >= 2 else 0.0
    if mode == "soft":
        if bad >= 2:
            return 0.75
        if bad == 1:
            return 0.25
        return 0.0
    raise ValueError(f"unsupported canary mode: {mode}")


def _inverse_vol(vol_row: pd.Series, assets: List[str]) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    inv = (1.0 / vol_row[assets].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() <= 0:
        inv[:] = 1.0
    return inv / inv.sum()


def run_canary_strategy(
    prices: pd.DataFrame,
    config: CanaryConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    vol63 = returns[ALL].rolling(63).std() * np.sqrt(252)
    mom21 = prices[ALL].pct_change(21, fill_method=None)
    mom63 = prices[ALL].pct_change(63, fill_method=None)
    mom126 = prices[ALL].pct_change(126, fill_method=None)
    mom252 = prices[ALL].pct_change(252, fill_method=None)
    sma200 = prices[RISKY].rolling(200).mean()
    schedule = schedule_flags(prices.index, config.execution_frequency)

    risk_off = overlay.risk_off.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0) if overlay is not None else pd.Series(0.0, index=prices.index)
    asset_bias = overlay.asset_bias.reindex(prices.index).fillna(0.0) if overlay is not None else pd.DataFrame(0.0, index=prices.index, columns=RISKY)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    for i, dt in enumerate(prices.index):
        if i < 252:
            weights.iloc[i] = current
            continue

        if bool(schedule.loc[dt]):
            score = _score_series(mom21, mom63, mom126, mom252, dt, config.score_mode)
            defensive_share = _canary_defensive_share(score, config.canary_mode)
            offensive_share = 1.0 - defensive_share

            llm_override = float(risk_off.loc[dt])
            offensive_share *= 1.0 - 0.65 * llm_override
            defensive_share = 1.0 - offensive_share

            offensive = [asset for asset in RISKY if score[asset] > 0]
            if config.use_trend:
                offensive = [asset for asset in offensive if prices.loc[dt, asset] > sma200.loc[dt, asset]]
            offensive = sorted(offensive, key=lambda asset: score[asset], reverse=True)[: config.top_risk]

            defensive = [asset for asset in DEFENSIVE_ASSETS if score[asset] > 0 or asset == "CASH"]
            defensive = sorted(defensive, key=lambda asset: score[asset], reverse=True)[: config.top_def]

            target = pd.Series(0.0, index=ALL)
            off_w = _inverse_vol(vol63.loc[dt], offensive)
            def_w = _inverse_vol(vol63.loc[dt], defensive)

            for asset, weight in off_w.items():
                target[asset] += offensive_share * weight
            for asset, weight in def_w.items():
                target[asset] += defensive_share * weight

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            if target[RISKY].sum() > 0 and float(bias_row.abs().sum()) > 0:
                multipliers = 1.0 + 0.15 * bias_row
                adjusted = target[RISKY] * multipliers
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum() * target[RISKY].sum()
                    target[RISKY] = adjusted

            target["CASH"] = max(0.0, 1.0 - target.drop("CASH").sum())
            target = target / target.sum()

            if float((target - current).abs().max()) > config.trade_band:
                current = target

        weights.iloc[i] = current

    return weights


def canary_walk_forward(
    prices: pd.DataFrame,
    candidates: List[CanaryConfig],
    rf: float,
    tx_cost: float,
    overlay: Optional[NarrativeOverlay] = None,
) -> Dict[str, object]:
    windows = []
    start = 0
    n = len(prices)
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        windows.append((start, start + TRAIN_DAYS, start + TRAIN_DAYS + TEST_DAYS))
        start += TEST_DAYS

    if not windows:
        return {
            "windows": [],
            "returns": pd.Series(dtype=float),
            "metrics": {
                "label": "Canary WFO Stitched OOS",
                "weights": pd.DataFrame(columns=ALL, dtype=float),
                "returns": pd.Series(dtype=float),
                "equity": pd.Series(dtype=float),
                "cagr": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "mdd": np.nan,
                "calmar": np.nan,
                "turnover": np.nan,
                "avg_cash": np.nan,
            },
        }

    stitched_returns: List[pd.Series] = []
    stitched_weights: List[pd.DataFrame] = []
    picked_rows: List[Dict[str, object]] = []

    for window_id, (train_start, train_end, test_end) in enumerate(windows, start=1):
        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        best_score = -np.inf
        best_config = candidates[0]
        for config in candidates:
            train_overlay = None
            if overlay is not None:
                train_overlay = NarrativeOverlay(
                    risk_off=overlay.risk_off.loc[train_prices.index],
                    asset_bias=overlay.asset_bias.loc[train_prices.index],
                )
            train_weights = run_canary_strategy(train_prices, config, overlay=train_overlay)
            train_metrics = performance_metrics(train_prices, train_weights, config.name, rf=rf, tx_cost=tx_cost)
            score = strategy_score(train_metrics)
            if score > best_score:
                best_score = score
                best_config = config

        combined = prices.iloc[train_start:test_end]
        combined_overlay = None
        if overlay is not None:
            combined_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[combined.index],
                asset_bias=overlay.asset_bias.loc[combined.index],
            )
        combined_weights = run_canary_strategy(combined, best_config, overlay=combined_overlay)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices, test_weights, best_config.name, rf=rf, tx_cost=tx_cost)

        picked_rows.append(
            {
                "window": window_id,
                "test_start": test_prices.index[0],
                "test_end": test_prices.index[-1],
                "picked": best_config.name,
                "cagr": float(test_metrics["cagr"]),
                "sharpe": float(test_metrics["sharpe"]),
                "mdd": float(test_metrics["mdd"]),
            }
        )
        stitched_returns.append(test_metrics["returns"])  # type: ignore[arg-type]
        stitched_weights.append(test_weights)

    stitched = pd.concat(stitched_returns).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    weights = pd.concat(stitched_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]
    weights = weights.reindex(stitched.index).ffill().fillna(0.0)

    equity = (1.0 + stitched).cumprod()
    years = len(stitched) / 252
    rf_daily = (1.0 + rf) ** (1 / 252) - 1.0
    excess = stitched - rf_daily
    vol = stitched.std() * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1.0).min()
    turnover = (weights.diff().abs().sum(axis=1) / 2.0).sum() / years

    return {
        "windows": picked_rows,
        "returns": stitched,
        "metrics": {
            "label": "Canary WFO Stitched OOS",
            "weights": weights,
            "returns": stitched,
            "equity": equity,
            "cagr": equity.iloc[-1] ** (1 / years) - 1.0,
            "vol": vol,
            "sharpe": excess.mean() * 252 / vol if vol > 0 else np.nan,
            "mdd": mdd,
            "calmar": (equity.iloc[-1] ** (1 / years) - 1.0) / abs(mdd) if mdd < 0 else np.nan,
            "turnover": turnover,
            "avg_cash": weights["CASH"].mean(),
        },
    }


def print_metric_line(label: str, metrics: Dict[str, object]) -> None:
    fmt = format_metrics(metrics)
    print(
        f"  {label:<24} CAGR {fmt['CAGR']:<8} "
        f"Sharpe {fmt['Sharpe']:<6} MaxDD {fmt['MaxDD']:<8} "
        f"Turn {fmt['AnnTurn']:<8} AvgCash {fmt['AvgCash']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Research lab for v9 vs canary momentum vs fixed composite.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--llm-override-file", help="Optional JSON narrative overlay file.")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource().fetch(args.start, end=args.end, refresh=args.refresh_cache)
    overlay = load_llm_overlay(args.llm_override_file, prices.index)

    v9_recommended = StrategyConfig(
        name="weekly_core85_tilt15",
        execution_frequency="WEEKLY",
        core_weight=0.85,
        tilt_weight=0.15,
        top_n=2,
        trade_band=0.08,
        trade_step=0.75,
        crash_floor=0.70,
    )

    canary_candidates = [
        CanaryConfig(
            name="canary_midlong_monthly",
            execution_frequency="MONTHLY",
            top_risk=2,
            top_def=2,
            trade_band=0.05,
            score_mode="midlong",
            canary_mode="fractional",
            use_trend=False,
        ),
        CanaryConfig(
            name="canary_blend13612_monthly",
            execution_frequency="MONTHLY",
            top_risk=2,
            top_def=2,
            trade_band=0.05,
            score_mode="blend13612",
            canary_mode="fractional",
            use_trend=False,
        ),
        CanaryConfig(
            name="canary_midlong_weekly",
            execution_frequency="WEEKLY",
            top_risk=3,
            top_def=2,
            trade_band=0.08,
            score_mode="midlong",
            canary_mode="fractional",
            use_trend=False,
        ),
    ]

    print("=" * 88)
    print("ALPHA ENGINE v10 — CANARY + DUAL MOMENTUM RESEARCH")
    print("=" * 88)
    print(f"Start date: {args.start}")
    print(f"Transaction cost: {tx_cost:.2%} per trade")
    print(f"Risk-free rate: {args.rf:.2%}")
    print("US sleeve: MON100 used directly as INR ETF return.")
    print("Composite benchmark: fixed strategic weights across all seven sleeves.")

    full_sample_results: List[Dict[str, object]] = []
    v9_weights = run_strategy(prices, v9_recommended, overlay=overlay)
    full_sample_results.append(performance_metrics(prices, v9_weights, v9_recommended.name, rf=args.rf, tx_cost=tx_cost))

    canary_full_sample: List[Dict[str, object]] = []
    for config in canary_candidates:
        weights = run_canary_strategy(prices, config, overlay=overlay)
        result = performance_metrics(prices, weights, config.name, rf=args.rf, tx_cost=tx_cost)
        canary_full_sample.append(result)
        full_sample_results.append(result)

    full_sample_results.extend(
        [
            fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),
            fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),
            performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost),
            performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost),
        ]
    )

    ordered = sorted(full_sample_results, key=lambda row: (float(row["sharpe"]), float(row["cagr"])), reverse=True)
    print_results_table("FULL SAMPLE", ordered)

    v9_wfo = walk_forward(prices, [v9_recommended], rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    canary_wfo = canary_walk_forward(prices, canary_candidates, rf=args.rf, tx_cost=tx_cost, overlay=overlay)

    oos_dates = v9_wfo["metrics"]["returns"].index if len(v9_wfo["metrics"]["returns"]) > 0 else canary_wfo["metrics"]["returns"].index  # type: ignore[index]
    oos_benchmarks = []
    if len(oos_dates) > 0:
        subset = prices.loc[oos_dates]
        oos_benchmarks = [
            fixed_weight_frame(subset, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),
            fixed_weight_frame(subset, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),
            *benchmark_oos_metrics(prices, oos_dates, rf=args.rf, tx_cost=tx_cost),
        ]

        print_results_table(
            "WALK-FORWARD OOS (2Y TRAIN / 6M TEST, STITCHED)",
            [v9_wfo["metrics"], canary_wfo["metrics"], *oos_benchmarks],  # type: ignore[list-item]
        )

    composite_full = next(row for row in full_sample_results if row["label"] == "Composite Fixed")
    eqwt_risky_full = next(row for row in full_sample_results if row["label"] == "EqWt Risky")
    best_canary = max(canary_full_sample, key=lambda row: (float(row["cagr"]), float(row["sharpe"])))

    print("\nResearch takeaways:")
    print_metric_line("v9 recommended", next(row for row in full_sample_results if row["label"] == v9_recommended.name))
    print_metric_line("best canary full sample", best_canary)
    print_metric_line("composite fixed", composite_full)
    print_metric_line("equal-weight risky", eqwt_risky_full)

    print("\nInterpretation:")
    print("  1. The canary family can lift CAGR versus v9 by concentrating harder into long-horizon winners.")
    print("  2. That extra return comes with a weaker Sharpe than the strategic composite and equal-weight risky benchmark.")
    print("  3. In this universe, simple long-only monthly momentum is useful, but it still does not justify the extreme claims often made for TAA systems.")
    print("  4. The realistic path to materially better results likely needs external macro/value data, not just fancier price-only transforms.")

    payload = {
        "full_sample": {
            row["label"]: {
                "cagr": row["cagr"],
                "vol": row["vol"],
                "sharpe": row["sharpe"],
                "mdd": row["mdd"],
                "calmar": row["calmar"],
                "turnover": row["turnover"],
                "avg_cash": row["avg_cash"],
            }
            for row in full_sample_results
        },
        "wfo": {
            "v9": {
                "cagr": v9_wfo["metrics"]["cagr"],
                "sharpe": v9_wfo["metrics"]["sharpe"],
                "mdd": v9_wfo["metrics"]["mdd"],
            },
            "canary": {
                "cagr": canary_wfo["metrics"]["cagr"],
                "sharpe": canary_wfo["metrics"]["sharpe"],
                "mdd": canary_wfo["metrics"]["mdd"],
            },
        },
        "composite_fixed_weights": COMPOSITE_FIXED_WEIGHTS,
    }
    RESEARCH_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESEARCH_RESULTS.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"\nSaved research summary to {RESEARCH_RESULTS}")


if __name__ == "__main__":
    main()
