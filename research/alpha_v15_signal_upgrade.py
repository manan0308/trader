#!/usr/bin/env python3
"""
ALPHA ENGINE v15 — SIGNAL UPGRADE CANDIDATE
===========================================

Research fork of v9 that tests a small set of externally-audited signal ideas
without changing production behavior.

Signals tested:
1. Low-vol breadth-filtered boost for MIDCAP / SMALLCAP.
2. India VIX capitulation tilt (post-panic, not peak-panic).
3. Silver deep rebound bonus.
4. Silver / gold ratio reversion tilt.
5. Smaller crash-time gold boost than production v9.

This script is intentionally fixed-rule and honest:
- same benchmark universe as production research
- 1-day lag via shared performance metrics
- 30 bps costs
- stitched walk-forward OOS using train/test windows only for warmup
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.significance_report import hac_difference_test, hac_mean_excess_test, iid_sharpe_test
from research.alpha_v11_macro_value_research import fetch_macro_panel
from research.alpha_v12_meta_ensemble import BASE_V9
from strategy.v9_engine import (
    ALL,
    RISKY,
    CACHE_DIR,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    DEFAULT_TX,
    NarrativeOverlay,
    StrategyConfig,
    YahooFinanceSource,
    benchmark_oos_metrics,
    benchmark_weights,
    breadth_risk_scale,
    build_features,
    crash_signal,
    format_metrics,
    inverse_vol_weights,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
)


RESULTS_CACHE = CACHE_DIR / "alpha_v15_signal_upgrade_results.json"


@dataclass(frozen=True)
class SignalUpgradeConfig:
    name: str
    base: StrategyConfig
    calm_breadth_min: int = 4
    calm_vol_ratio: float = 0.70
    calm_boost: float = 1.15
    vix_peak_threshold: float = 3.0
    vix_drop_threshold: float = 1.0
    vix_tilt_bonus: float = 0.05
    enable_vix_capitulation: bool = True
    silver_deep_rsi: float = 30.0
    silver_deep_dd20: float = -0.10
    silver_deep_bonus: float = 0.05
    silver_gold_ratio_threshold: float = -0.10
    silver_gold_bonus: float = 0.04
    gold_ratio_penalty: float = 0.02
    crash_gold_boost: float = 0.05


def build_signal_features(prices: pd.DataFrame, macro: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    feats = dict(build_features(prices))
    returns = pd.DataFrame(feats["returns"]).reindex(prices.index)
    asset_vol252 = returns[RISKY].rolling(252).std() * np.sqrt(252.0)
    asset_vol63 = pd.DataFrame(feats["vol63"]).reindex(prices.index)
    asset_vol63_252 = asset_vol63 / asset_vol252.replace(0, np.nan)

    ratio = prices["SILVER"] / prices["GOLD"]
    silver_gold_ratio_mom63 = ratio.pct_change(63, fill_method=None)

    aligned_macro = macro.reindex(prices.index).ffill()
    vix = aligned_macro["INDIAVIX"] if "INDIAVIX" in aligned_macro.columns else pd.Series(np.nan, index=prices.index)
    vix_mean = vix.rolling(252).mean()
    vix_std = vix.rolling(252).std()
    vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
    vix_peak_10 = vix_z.rolling(10).max()

    feats["asset_vol252"] = asset_vol252
    feats["asset_vol63_252"] = asset_vol63_252
    feats["silver_gold_ratio_mom63"] = silver_gold_ratio_mom63
    feats["vix_z"] = vix_z
    feats["vix_peak_10"] = vix_peak_10
    return feats


def run_signal_upgrade_strategy(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: SignalUpgradeConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    feats = build_signal_features(prices, macro)
    schedule = schedule_flags(prices.index, config.base.execution_frequency)
    override = (
        overlay.risk_off.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if overlay is not None
        else pd.Series(0.0, index=prices.index)
    )
    asset_bias = (
        overlay.asset_bias.reindex(prices.index).fillna(0.0)
        if overlay is not None
        else pd.DataFrame(0.0, index=prices.index, columns=RISKY)
    )

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = pd.DataFrame(feats["sma50"])
    sma200 = pd.DataFrame(feats["sma200"])
    mom63 = pd.DataFrame(feats["mom63"])
    mom126 = pd.DataFrame(feats["mom126"])
    mom252 = pd.DataFrame(feats["mom252"])
    vol63 = pd.DataFrame(feats["vol63"])
    rsi14 = pd.DataFrame(feats["rsi14"])
    asset_dd20 = pd.DataFrame(feats["asset_dd20"])
    asset_vol_ratio = pd.DataFrame(feats["asset_vol_ratio"])
    breakout252 = pd.DataFrame(feats["breakout252"])
    cash126 = pd.Series(feats["cash126"])
    asset_vol63_252 = pd.DataFrame(feats["asset_vol63_252"])
    silver_gold_ratio_mom63 = pd.Series(feats["silver_gold_ratio_mom63"])
    vix_z = pd.Series(feats["vix_z"])
    vix_peak_10 = pd.Series(feats["vix_peak_10"])

    for i, dt in enumerate(prices.index):
        if i < 252:
            weights.iloc[i] = current
            continue

        trend = prices.loc[dt, RISKY] > sma200.loc[dt]
        strong = trend & (prices.loc[dt, RISKY] > sma50.loc[dt]) & (mom126.loc[dt] > cash126.loc[dt])
        breadth = int(strong.sum())
        crash = crash_signal(feats, dt)

        if bool(schedule.loc[dt]):
            risk_scale = breadth_risk_scale(breadth)
            if crash:
                risk_scale = min(risk_scale, config.base.crash_floor)

            llm_override = float(override.loc[dt])
            risk_scale = min(risk_scale, 1.0 - 0.65 * llm_override)

            score = 0.20 * (mom63.loc[dt] / vol63.loc[dt].replace(0, np.nan)) + 0.35 * mom126.loc[dt] + 0.45 * mom252.loc[dt]
            score = score.replace([np.inf, -np.inf], np.nan).fillna(-999.0).sort_values(ascending=False)

            rebound_setup = trend & (asset_dd20.loc[dt] <= -0.08) & (rsi14.loc[dt] <= 45.0)
            score = score + rebound_setup.astype(float) * 0.04

            if bool(trend.get("GOLD", False)) and float(asset_vol_ratio.loc[dt].get("GOLD", 0.0) or 0.0) > 1.35:
                score.loc["GOLD"] = float(score.get("GOLD", -999.0)) + 0.03
            if bool(trend.get("SILVER", False)) and bool(breakout252.loc[dt].get("SILVER", False)):
                score.loc["SILVER"] = float(score.get("SILVER", -999.0)) + 0.03

            silver_deep_rebound = (
                bool(trend.get("SILVER", False))
                and float(rsi14.loc[dt].get("SILVER", 50.0)) < config.silver_deep_rsi
                and float(asset_dd20.loc[dt].get("SILVER", 0.0)) < config.silver_deep_dd20
            )
            if silver_deep_rebound:
                score.loc["SILVER"] = float(score.get("SILVER", -999.0)) + config.silver_deep_bonus

            ratio_mom = float(silver_gold_ratio_mom63.loc[dt]) if pd.notna(silver_gold_ratio_mom63.loc[dt]) else np.nan
            if np.isfinite(ratio_mom) and ratio_mom < config.silver_gold_ratio_threshold:
                score.loc["SILVER"] = float(score.get("SILVER", -999.0)) + config.silver_gold_bonus
                if "GOLD" in score.index:
                    score.loc["GOLD"] = float(score.get("GOLD", -999.0)) - config.gold_ratio_penalty

            if config.enable_vix_capitulation:
                vz = float(vix_z.loc[dt]) if pd.notna(vix_z.loc[dt]) else np.nan
                peak10 = float(vix_peak_10.loc[dt]) if pd.notna(vix_peak_10.loc[dt]) else np.nan
                vix_capitulation = (
                    np.isfinite(vz)
                    and np.isfinite(peak10)
                    and peak10 > config.vix_peak_threshold
                    and vz < peak10 - config.vix_drop_threshold
                    and vz > 0.0
                )
                if vix_capitulation:
                    for asset in ("MIDCAP", "SMALLCAP"):
                        if asset in score.index and bool(trend.get(asset, False)):
                            score.loc[asset] = float(score.get(asset, -999.0)) + config.vix_tilt_bonus

            selected = [asset for asset in score.index if bool(trend.get(asset, False))][: config.base.top_n]

            target = pd.Series(0.0, index=ALL)
            core_w = inverse_vol_weights(vol63.loc[dt], RISKY)

            if breadth >= config.calm_breadth_min:
                core_w = core_w.copy()
                for asset in ("MIDCAP", "SMALLCAP"):
                    ratio_today = float(asset_vol63_252.loc[dt].get(asset, np.nan))
                    if np.isfinite(ratio_today) and ratio_today < config.calm_vol_ratio:
                        core_w.loc[asset] *= config.calm_boost
                core_w = core_w / core_w.sum()

            for asset, weight in core_w.items():
                target[asset] += risk_scale * config.base.core_weight * weight

            if selected and config.base.tilt_weight > 0:
                tilt_scores = score.loc[selected].clip(lower=0.0)
                if tilt_scores.sum() <= 0:
                    tilt_scores[:] = 1.0
                tilt_scores = tilt_scores / tilt_scores.sum()
                for asset, weight in tilt_scores.items():
                    target[asset] += risk_scale * config.base.tilt_weight * weight

            if crash and bool(trend.get("GOLD", False)):
                target["GOLD"] += min(config.crash_gold_boost, 1.0 - target[RISKY].sum())

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            if target[RISKY].sum() > 0 and float(bias_row.abs().sum()) > 0:
                multipliers = 1.0 + 0.15 * bias_row
                adjusted = target[RISKY] * multipliers
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum() * target[RISKY].sum()
                    target[RISKY] = adjusted

            target["CASH"] = max(0.0, 1.0 - target[RISKY].sum())
            target = target / target.sum()

            proposal = current * (1.0 - config.base.trade_step) + target * config.base.trade_step
            proposal["CASH"] = max(0.0, 1.0 - proposal[RISKY].sum())
            proposal = proposal / proposal.sum()

            if float((proposal - current).abs().max()) > config.base.trade_band:
                current = proposal

        if crash and current[RISKY].sum() > 0.80:
            current = current.copy()
            current[RISKY] *= 0.90
            current["CASH"] = 1.0 - current[RISKY].sum()

        llm_live_risk = float(override.loc[dt])
        if llm_live_risk > 0.15 and current[RISKY].sum() > 0:
            min_cash = float(np.clip(0.20 + 0.45 * llm_live_risk, 0.20, 0.70))
            max_risky = 1.0 - min_cash
            risky_sum = float(current[RISKY].sum())
            if risky_sum > max_risky:
                current = current.copy()
                current[RISKY] *= max_risky / risky_sum
                current["CASH"] = 1.0 - current[RISKY].sum()

        weights.iloc[i] = current

    return weights


def fixed_config_wfo(
    prices: pd.DataFrame,
    runner: Callable[[pd.DataFrame], pd.DataFrame],
    label: str,
    rf: float,
    tx_cost: float,
    train_days: int = 756,
    test_days: int = 126,
) -> Dict[str, object]:
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    if not windows:
        empty_returns = pd.Series(dtype=float)
        empty_weights = pd.DataFrame(columns=ALL, dtype=float)
        return {
            "windows": [],
            "metrics": {
                "label": label,
                "weights": empty_weights,
                "returns": empty_returns,
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
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        combined_prices = prices.iloc[start_i:end_i]
        test_prices = prices.iloc[mid_i:end_i]
        combined_weights = runner(combined_prices)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices, test_weights, label, rf=rf, tx_cost=tx_cost)
        rows.append(
            {
                "window": window_id,
                "test_start": test_prices.index[0],
                "test_end": test_prices.index[-1],
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
    years = len(stitched) / 252.0
    rf_daily = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    excess = stitched - rf_daily
    vol = stitched.std() * np.sqrt(252.0)
    mdd = (equity / equity.cummax() - 1.0).min()
    turnover = (weights.diff().abs().sum(axis=1) / 2.0).sum() / years

    return {
        "windows": rows,
        "metrics": {
            "label": label,
            "weights": weights,
            "returns": stitched,
            "equity": equity,
            "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
            "vol": vol,
            "sharpe": excess.mean() * 252.0 / vol if vol > 0 else np.nan,
            "mdd": mdd,
            "calmar": (equity.iloc[-1] ** (1.0 / years) - 1.0) / abs(mdd) if mdd < 0 else np.nan,
            "turnover": turnover,
            "avg_cash": weights["CASH"].mean(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test signal-upgraded v9 candidates against production v9.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000.0
    prices = YahooFinanceSource(universe_mode="benchmark").fetch(args.start, end=args.end, refresh=args.refresh_cache)
    macro = fetch_macro_panel(args.start, refresh=args.refresh_cache).reindex(prices.index).ffill()

    candidate_all = SignalUpgradeConfig(name="v15_signal_upgrade_all", base=BASE_V9)
    candidate_no_vix = replace(candidate_all, name="v15_signal_upgrade_no_vix", enable_vix_capitulation=False)

    runners = {
        BASE_V9.name: lambda frame: run_strategy(frame, BASE_V9, overlay=None),
        candidate_all.name: lambda frame: run_signal_upgrade_strategy(frame, macro.reindex(frame.index).ffill(), candidate_all, overlay=None),
        candidate_no_vix.name: lambda frame: run_signal_upgrade_strategy(frame, macro.reindex(frame.index).ffill(), candidate_no_vix, overlay=None),
    }

    full_sample: List[Dict[str, object]] = []
    for label, runner in runners.items():
        weights = runner(prices)
        full_sample.append(performance_metrics(prices, weights, label, rf=args.rf, tx_cost=tx_cost))

    print_results_table("FULL SAMPLE", full_sample)

    wfo_results = {
        label: fixed_config_wfo(prices, runner, label, rf=args.rf, tx_cost=tx_cost)
        for label, runner in runners.items()
    }
    oos_rows = [payload["metrics"] for payload in wfo_results.values()]
    base_dates = pd.DatetimeIndex(wfo_results[BASE_V9.name]["metrics"]["returns"].index)
    oos_benchmarks = benchmark_oos_metrics(prices, base_dates, rf=args.rf, tx_cost=tx_cost) if len(base_dates) > 0 else []
    print_results_table("WALK-FORWARD OOS (FIXED CONFIG, STITCHED)", [*oos_rows, *oos_benchmarks])  # type: ignore[list-item]

    base_returns = pd.Series(wfo_results[BASE_V9.name]["metrics"]["returns"])
    pairwise = {}
    for label in (candidate_all.name, candidate_no_vix.name):
        candidate_returns = pd.Series(wfo_results[label]["metrics"]["returns"])
        pairwise[f"{label}_minus_v9"] = hac_difference_test(candidate_returns, base_returns)
        pairwise[f"{label}_mean_tests"] = {
            **iid_sharpe_test(candidate_returns, args.rf),
            **hac_mean_excess_test(candidate_returns, args.rf),
        }

    payload = {
        "start": args.start,
        "end": args.end,
        "tx_cost": tx_cost,
        "full_sample": {
            row["label"]: {
                **{k: float(v) for k, v in row.items() if k not in {"label", "weights", "returns", "equity"}},
                "pretty": format_metrics(row),
            }
            for row in full_sample
        },
        "wfo": {
            label: {
                "windows": result["windows"],
                "metrics": {
                    **{
                        k: float(v)
                        for k, v in result["metrics"].items()
                        if k not in {"label", "weights", "returns", "equity"}
                    },
                    "pretty": format_metrics(result["metrics"]),
                },
            }
            for label, result in wfo_results.items()
        },
        "pairwise": pairwise,
    }
    RESULTS_CACHE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {RESULTS_CACHE}")


if __name__ == "__main__":
    main()
