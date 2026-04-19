#!/usr/bin/env python3
"""
ALPHA ENGINE v21 - PRICE/VOLUME ROTATION
========================================

Two simple ETF-universe challengers motivated by the April 2026 tape:

1. OBV 5-day acceleration (z-scored) -> top 2 assets daily
2. 10-day momentum * 20-day relative volume -> top 2 assets daily

These are intentionally simple and fully invested. They are not production
promotions by default; the goal is to measure whether the recent burst of
performance survives the broader ETF sample after realistic India delivery
costs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
from research.alpha_v12_meta_ensemble import BASE_V9
from research.alpha_v18_agile_rotation import ETF_SYMBOL_MAP
from strategy.v9_engine import (
    ALL,
    CACHE_DIR,
    DEFAULT_RF,
    DEFAULT_TX,
    WARMUP_DAYS,
    format_metrics,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
)


DEFAULT_ETF = Path("/Users/mananagarwal/Desktop/historical data/dhan/etfs/all_daily_fy24_fy26_aligned.parquet")
DEFAULT_RECENT = CACHE_DIR / "dhan_recent_2026_03_18_2026_04_18.csv"
DEFAULT_OUTPUT = CACHE_DIR / "alpha_v21_price_volume_rotation_results.json"
RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]


def load_merged_etf_history(parquet_path: Path, recent_csv: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = pd.read_parquet(parquet_path)
    hist["date"] = pd.to_datetime(hist["date"])
    frames = [hist]
    if recent_csv and recent_csv.exists():
        recent = pd.read_csv(recent_csv, parse_dates=["date"])
        frames.append(recent)

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
    )
    merged = merged[merged["symbol"].isin(ETF_SYMBOL_MAP)].copy()
    merged["asset"] = merged["symbol"].map(ETF_SYMBOL_MAP)

    prices = (
        merged.pivot(index="date", columns="asset", values="close")
        .sort_index()
        .reindex(columns=ALL)
        .dropna()
    )
    volume = (
        merged.pivot(index="date", columns="asset", values="volume")
        .sort_index()
        .reindex(index=prices.index, columns=ALL)
        .fillna(0.0)
        .astype(float)
    )
    return prices, volume


def zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window).mean()
    std = frame.rolling(window).std().replace(0.0, np.nan)
    return (frame - mean) / std


def build_obv5_scores(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    returns = prices[RISKY].pct_change(fill_method=None)
    obv = (np.sign(returns.fillna(0.0)) * volume[RISKY]).fillna(0.0).cumsum()
    obv5 = obv.diff(5)
    return zscore(obv5, 60)


def build_mom10_x_vr20_scores(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    mom10 = prices[RISKY].pct_change(10)
    volume_ratio20 = volume[RISKY] / volume[RISKY].rolling(20).median().replace(0.0, np.nan)
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
        for asset in picks:
            targets.loc[dt, asset] = 1.0 / len(picks)
    return targets


def apply_rebalance_policy(
    targets: pd.DataFrame,
    *,
    frequency: str,
    trade_band: float,
    trade_step: float,
) -> pd.DataFrame:
    if frequency not in {"DAILY", "WEEKLY"}:
        raise ValueError(f"unsupported v21 frequency: {frequency}")

    if frequency == "DAILY":
        schedule = pd.Series(True, index=targets.index)
    else:
        schedule = schedule_flags(targets.index, "WEEKLY")

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
            if float(proposal.sum()) > 0:
                proposal = proposal / proposal.sum()
            else:
                proposal = target
            if float((proposal - current).abs().max()) > trade_band:
                current = proposal
        weights.loc[dt] = current
    return weights


def strip_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    keep = {key: value for key, value in metrics.items() if key not in {"label", "weights", "returns", "equity"}}
    keep["pretty"] = format_metrics(metrics)
    return keep


def blend_weights(left: pd.DataFrame, right: pd.DataFrame, left_weight: float) -> pd.DataFrame:
    aligned_left = left.reindex(columns=ALL).fillna(0.0)
    aligned_right = right.reindex(index=aligned_left.index, columns=ALL).fillna(0.0)
    blended = left_weight * aligned_left + (1.0 - left_weight) * aligned_right
    row_sum = blended.sum(axis=1).replace(0.0, np.nan)
    return blended.div(row_sum, axis=0).fillna(0.0)


def window_metrics(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    label: str,
    *,
    start: Optional[str],
    rf: float,
    tx_cost: float,
    cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
) -> Dict[str, object]:
    if start:
        prices = prices.loc[start:]
        weights = weights.loc[prices.index]
    metrics = performance_metrics(
        prices,
        weights,
        label,
        rf=rf,
        tx_cost=tx_cost,
        cost_model=cost_model,
        base_value=base_value,
    )
    return strip_metrics(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest price/volume ETF rotation challengers.")
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--recent-csv", default=str(DEFAULT_RECENT))
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--cost-model", choices=["flat", "india_delivery"], default="india_delivery")
    parser.add_argument("--base-value", type=float, default=1_000_000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    parquet_path = Path(args.etf_parquet).expanduser().resolve()
    recent_csv = Path(args.recent_csv).expanduser().resolve() if args.recent_csv else None
    output_path = Path(args.output).expanduser().resolve()

    tx_cost = args.tx_bps / 10_000.0
    cost_model = resolve_cost_model(args.cost_model)

    prices, volume = load_merged_etf_history(parquet_path, recent_csv)
    oos_start = prices.index[min(252, len(prices) - 1)].strftime("%Y-%m-%d")

    v9 = run_strategy(prices, BASE_V9, overlay=None).reindex(columns=ALL).fillna(0.0)
    obv5_targets = build_top_n_targets(build_obv5_scores(prices, volume), top_n=2)
    mom10_x_vr20_targets = build_top_n_targets(build_mom10_x_vr20_scores(prices, volume), top_n=2)
    obv5_weekly_s050_b020 = apply_rebalance_policy(obv5_targets, frequency="WEEKLY", trade_band=0.20, trade_step=0.50)
    mom10_x_vr20_weekly_s050_b020 = apply_rebalance_policy(
        mom10_x_vr20_targets,
        frequency="WEEKLY",
        trade_band=0.20,
        trade_step=0.50,
    )

    strategies = {
        "v9_quant": v9,
        "v21_obv5_top2_daily": apply_rebalance_policy(obv5_targets, frequency="DAILY", trade_band=0.0, trade_step=1.0),
        "v21_obv5_top2_weekly": apply_rebalance_policy(obv5_targets, frequency="WEEKLY", trade_band=0.0, trade_step=1.0),
        "v21_obv5_top2_weekly_s050_b010": apply_rebalance_policy(obv5_targets, frequency="WEEKLY", trade_band=0.10, trade_step=0.50),
        "v21_obv5_top2_weekly_s050_b020": obv5_weekly_s050_b020,
        "v21_obv5_top2_weekly_s075_b015": apply_rebalance_policy(obv5_targets, frequency="WEEKLY", trade_band=0.15, trade_step=0.75),
        "v21_mom10_x_vr20_top2_daily": apply_rebalance_policy(mom10_x_vr20_targets, frequency="DAILY", trade_band=0.0, trade_step=1.0),
        "v21_mom10_x_vr20_top2_weekly": apply_rebalance_policy(mom10_x_vr20_targets, frequency="WEEKLY", trade_band=0.0, trade_step=1.0),
        "v21_mom10_x_vr20_top2_weekly_s050_b010": apply_rebalance_policy(mom10_x_vr20_targets, frequency="WEEKLY", trade_band=0.10, trade_step=0.50),
        "v21_mom10_x_vr20_top2_weekly_s050_b020": mom10_x_vr20_weekly_s050_b020,
        "v21_mom10_x_vr20_top2_weekly_s075_b015": apply_rebalance_policy(mom10_x_vr20_targets, frequency="WEEKLY", trade_band=0.15, trade_step=0.75),
        "v21_blend_90v9_10momvol": blend_weights(v9, mom10_x_vr20_weekly_s050_b020, 0.90),
        "v21_blend_80v9_20momvol": blend_weights(v9, mom10_x_vr20_weekly_s050_b020, 0.80),
        "v21_blend_70v9_30momvol": blend_weights(v9, mom10_x_vr20_weekly_s050_b020, 0.70),
        "v21_blend_60v9_40momvol": blend_weights(v9, mom10_x_vr20_weekly_s050_b020, 0.60),
        "v21_blend_90v9_10obv": blend_weights(v9, obv5_weekly_s050_b020, 0.90),
        "v21_blend_80v9_20obv": blend_weights(v9, obv5_weekly_s050_b020, 0.80),
        "v21_blend_70v9_30obv": blend_weights(v9, obv5_weekly_s050_b020, 0.70),
        "v21_blend_60v9_40obv": blend_weights(v9, obv5_weekly_s050_b020, 0.60),
    }

    full_rows = []
    oos_rows = []
    payload_rows: Dict[str, object] = {}
    for label, weights in strategies.items():
        full_metrics = window_metrics(
            prices,
            weights,
            label,
            start=None,
            rf=args.rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=args.base_value,
        )
        oos_metrics = window_metrics(
            prices,
            weights,
            label,
            start=oos_start,
            rf=args.rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=args.base_value,
        )
        full_rows.append({"label": label, **full_metrics})
        oos_rows.append({"label": label, **oos_metrics})
        payload_rows[label] = {
            "full_sample": full_metrics,
            "stitched_oos_after_252d": oos_metrics,
            "latest_weights": {asset: float(weights.iloc[-1][asset]) for asset in weights.columns},
        }

    print_results_table("FULL SAMPLE - ETF PRICE/VOLUME CHALLENGERS", full_rows)
    print_results_table(f"OOS AFTER 252-DAY WARMUP - start {oos_start}", oos_rows)

    output = {
        "etf_parquet": str(parquet_path),
        "recent_csv": str(recent_csv) if recent_csv else None,
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "rows": int(len(prices)),
        "tx_cost": tx_cost,
        "cost_model": args.cost_model,
        "base_value": args.base_value,
        "oos_start_after_252d": oos_start,
        "strategies": payload_rows,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
