#!/usr/bin/env python3
"""
ALPHA ENGINE v14 — SPARSE SLEEVE META
=====================================

Research goal:
- Keep the strong parts of v12.
- Enforce sparsity at the sleeve level rather than the asset level.
- Keep Composite Fixed as a permanent anchor.
- Allow only the best 1-2 tactical sleeves to survive each rebalance.

Why this branch exists:
- v13 showed that asset-level sparsity can work, but it adds another layer of
  name-level churn and selection risk.
- This version concentrates one layer higher in the stack: composite is always
  present, while v9 / macro / canary compete for limited tactical slots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.v9_engine import (
    ALL,
    CACHE_DIR,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    DEFAULT_TX,
    NarrativeOverlay,
    YahooFinanceSource,
    benchmark_oos_metrics,
    benchmark_weights,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
)
from research.alpha_v10_canary_research import COMPOSITE_FIXED_WEIGHTS
from research.alpha_v11_macro_value_research import fetch_macro_panel, parametric_v9_wfo
from research.alpha_v12_meta_ensemble import (
    BASE_V9,
    MetaConfig,
    build_meta_inputs,
    build_sleeves,
    current_meta_target,
    fixed_weight_frame,
    meta_candidates,
    meta_walk_forward,
    run_meta_strategy,
)
from research.alpha_v13_sparse_meta import fixed_sparse_walk_forward, risk_preserving_candidates as v13_candidates


RESULTS_CACHE = CACHE_DIR / "alpha_v14_sparse_sleeves_results.json"
TACTICAL_SLEEVES = ["v9", "macro", "canary"]
ALL_SLEEVES = ["composite", *TACTICAL_SLEEVES]


@dataclass(frozen=True)
class SleeveSparseConfig:
    name: str
    base_meta: str
    tactical_top_k: int
    selection_frequency: str
    rank_buffer: int
    min_tactical_weight: float
    trade_band: float
    trade_mix: float


def active_tactical_sleeves(meta_weights: pd.DataFrame) -> float:
    return float((meta_weights[TACTICAL_SLEEVES] > 0.001).sum(axis=1).mean())


def resolve_meta_config(name: str) -> MetaConfig:
    mapping = {cfg.name: cfg for cfg in meta_candidates()}
    if name not in mapping:
        raise KeyError(f"Unknown base meta config: {name}")
    return mapping[name]


def sleeve_sparse_candidates() -> List[SleeveSparseConfig]:
    return [
        SleeveSparseConfig(
            name="sleeve_top1_slow",
            base_meta="meta_126252_direct",
            tactical_top_k=1,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_tactical_weight=0.08,
            trade_band=0.04,
            trade_mix=1.0,
        ),
        SleeveSparseConfig(
            name="sleeve_top2_slow",
            base_meta="meta_126252_direct",
            tactical_top_k=2,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_tactical_weight=0.06,
            trade_band=0.04,
            trade_mix=1.0,
        ),
        SleeveSparseConfig(
            name="sleeve_top1_smooth",
            base_meta="meta_63126_smooth",
            tactical_top_k=1,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_tactical_weight=0.08,
            trade_band=0.04,
            trade_mix=1.0,
        ),
        SleeveSparseConfig(
            name="sleeve_top2_smooth",
            base_meta="meta_63126_smooth",
            tactical_top_k=2,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_tactical_weight=0.06,
            trade_band=0.04,
            trade_mix=1.0,
        ),
    ]


def selection_schedule(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    if frequency == "MONTHLY":
        bucket = index.to_series().dt.to_period("M").astype(str)
    elif frequency == "WEEKLY":
        bucket = index.to_series().dt.strftime("%Y-%U")
    else:
        raise ValueError(f"unsupported frequency: {frequency}")
    return bucket.ne(bucket.shift(-1)).fillna(True)


def pick_tactical_sleeves(
    tactical_weights: pd.Series,
    previous: List[str],
    config: SleeveSparseConfig,
) -> List[str]:
    ranked = tactical_weights.sort_values(ascending=False)
    ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)

    keep: List[str] = []
    for sleeve in previous:
        if sleeve not in ranked.index:
            continue
        if int(ranks[sleeve]) <= config.tactical_top_k + config.rank_buffer or float(tactical_weights[sleeve]) >= config.min_tactical_weight:
            keep.append(sleeve)

    picks = list(keep)
    for sleeve in ranked.index:
        if sleeve in picks:
            continue
        if float(tactical_weights[sleeve]) <= 0.001:
            continue
        picks.append(sleeve)
        if len(picks) >= config.tactical_top_k:
            break

    if not picks:
        picks = ranked.head(config.tactical_top_k).index.tolist()
    return picks[: config.tactical_top_k]


def project_sparse_sleeves(
    dense_target: pd.Series,
    selected_tactical: List[str],
) -> pd.Series:
    sparse = pd.Series(0.0, index=ALL_SLEEVES, dtype=float)
    composite_weight = float(dense_target.get("composite", 0.0))
    sparse["composite"] = composite_weight

    for sleeve in selected_tactical:
        sparse[sleeve] = float(dense_target.get(sleeve, 0.0))

    dropped = float(dense_target[TACTICAL_SLEEVES].sum()) - float(sparse[TACTICAL_SLEEVES].sum())
    sparse["composite"] += max(0.0, dropped)
    if float(sparse.sum()) <= 0:
        sparse["composite"] = 1.0
    return sparse / float(sparse.sum())


def run_sparse_sleeve_strategy(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: SleeveSparseConfig,
    overlay: Optional[NarrativeOverlay] = None,
    tx_cost: float = DEFAULT_TX,
    return_meta: bool = False,
):
    base_cfg = resolve_meta_config(config.base_meta)
    sleeves = build_sleeves(prices, macro, overlay=overlay)
    meta_inputs = build_meta_inputs(prices, macro, sleeves, tx_cost=tx_cost)

    sleeve_returns = meta_inputs["returns"]  # type: ignore[assignment]
    sleeve_weights = meta_inputs["weights"]  # type: ignore[assignment]
    breadth = meta_inputs["breadth"]  # type: ignore[assignment]
    vix_ratio = meta_inputs["vix_ratio"]  # type: ignore[assignment]
    schedule = selection_schedule(sleeve_returns.index, config.selection_frequency)

    risk_off = (
        overlay.risk_off.reindex(sleeve_returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if overlay is not None
        else pd.Series(0.0, index=sleeve_returns.index)
    )

    current = pd.Series({"composite": 0.60, "v9": 0.25, "macro": 0.15, "canary": 0.0}, dtype=float)
    meta_weights = pd.DataFrame(0.0, index=sleeve_returns.index, columns=current.index)
    selected_tactical = ["v9", "macro"][: config.tactical_top_k]

    min_history = max(base_cfg.sleeve_mom_slow, 252)
    for i, dt in enumerate(sleeve_returns.index):
        if i < min_history:
            meta_weights.loc[dt] = current
            continue

        if bool(schedule.loc[dt]):
            dense_target = current_meta_target(
                dt=dt,
                subset=sleeve_returns.iloc[: i + 1],
                breadth=breadth,
                vix_ratio=vix_ratio,
                config=base_cfg,
                llm_risk=float(risk_off.loc[dt]),
            )
            selected_tactical = pick_tactical_sleeves(dense_target[TACTICAL_SLEEVES], selected_tactical, config)
            sparse_target = project_sparse_sleeves(dense_target, selected_tactical)

            proposal = config.trade_mix * sparse_target + (1.0 - config.trade_mix) * current
            for sleeve in TACTICAL_SLEEVES:
                if sleeve not in selected_tactical:
                    proposal[sleeve] = 0.0
            proposal["composite"] = 1.0 - float(proposal[TACTICAL_SLEEVES].sum())
            proposal = proposal / float(proposal.sum())

            if float((proposal - current).abs().max()) > config.trade_band:
                current = proposal

        meta_weights.loc[dt] = current

    asset_weights = pd.DataFrame(0.0, index=sleeve_returns.index, columns=ALL)
    for sleeve_name, sleeve_frame in sleeve_weights.items():
        asset_weights = asset_weights.add(
            meta_weights[sleeve_name].values.reshape(-1, 1) * sleeve_frame[ALL].values,
            fill_value=0.0,
        )
    asset_weights = asset_weights.div(asset_weights.sum(axis=1), axis=0).fillna(0.0)

    if return_meta:
        return asset_weights, meta_weights
    return asset_weights


def fixed_sleeve_sparse_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: SleeveSparseConfig,
    rf: float,
    tx_cost: float,
    overlay: Optional[NarrativeOverlay] = None,
    train_days: int = 756,
    test_days: int = 126,
) -> Dict[str, object]:
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    stitched_weights: List[pd.DataFrame] = []
    stitched_meta: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        combined_prices = prices.iloc[start_i:end_i]
        combined_macro = macro.reindex(combined_prices.index).ffill()
        combined_overlay = None
        if overlay is not None:
            combined_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[combined_prices.index],
                asset_bias=overlay.asset_bias.loc[combined_prices.index],
            )
        combined_weights, combined_meta = run_sparse_sleeve_strategy(
            combined_prices,
            combined_macro,
            config,
            overlay=combined_overlay,
            tx_cost=tx_cost,
            return_meta=True,
        )
        test_prices = prices.iloc[mid_i:end_i]
        test_weights = combined_weights.loc[test_prices.index]
        test_meta = combined_meta.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices.loc[test_weights.index], test_weights, config.name, rf=rf, tx_cost=tx_cost)
        stitched_weights.append(test_weights)
        stitched_meta.append(test_meta)
        rows.append(
            {
                "window": window_id,
                "picked": config.name,
                "test_start": test_prices.index[0],
                "test_end": test_prices.index[-1],
                "cagr": float(test_metrics["cagr"]),
                "sharpe": float(test_metrics["sharpe"]),
                "mdd": float(test_metrics["mdd"]),
                "turnover": float(test_metrics["turnover"]),
            }
        )

    weights = pd.concat(stitched_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]
    meta_weights = pd.concat(stitched_meta).sort_index()
    meta_weights = meta_weights[~meta_weights.index.duplicated(keep="last")]

    metrics = performance_metrics(prices.loc[weights.index], weights, f"{config.name} WFO Fixed OOS", rf=rf, tx_cost=tx_cost)
    metrics["avg_active_sleeves"] = active_tactical_sleeves(meta_weights)
    return {
        "windows": rows,
        "returns": metrics["returns"],  # type: ignore[index]
        "metrics": metrics,
        "meta_weights": meta_weights,
    }


def sleeve_sparse_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    candidates: List[SleeveSparseConfig],
    rf: float,
    tx_cost: float,
    overlay: Optional[NarrativeOverlay] = None,
    train_days: int = 756,
    test_days: int = 126,
) -> Dict[str, object]:
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    stitched_weights: List[pd.DataFrame] = []
    stitched_meta: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        train_prices = prices.iloc[start_i:mid_i]
        train_macro = macro.reindex(train_prices.index).ffill()
        train_overlay = None
        if overlay is not None:
            train_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[train_prices.index],
                asset_bias=overlay.asset_bias.loc[train_prices.index],
            )

        best_cfg = candidates[0]
        best_score = -np.inf
        for cfg in candidates:
            train_weights, train_meta = run_sparse_sleeve_strategy(
                train_prices,
                train_macro,
                cfg,
                overlay=train_overlay,
                tx_cost=tx_cost,
                return_meta=True,
            )
            train_metrics = performance_metrics(train_prices.loc[train_weights.index], train_weights, cfg.name, rf=rf, tx_cost=tx_cost)
            score = float(train_metrics["sharpe"]) + 0.30 * float(train_metrics["calmar"]) - 0.06 * float(train_metrics["turnover"])
            if score > best_score:
                best_score = score
                best_cfg = cfg

        combined_prices = prices.iloc[start_i:end_i]
        combined_macro = macro.reindex(combined_prices.index).ffill()
        combined_overlay = None
        if overlay is not None:
            combined_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[combined_prices.index],
                asset_bias=overlay.asset_bias.loc[combined_prices.index],
            )
        combined_weights, combined_meta = run_sparse_sleeve_strategy(
            combined_prices,
            combined_macro,
            best_cfg,
            overlay=combined_overlay,
            tx_cost=tx_cost,
            return_meta=True,
        )

        test_prices = prices.iloc[mid_i:end_i]
        test_weights = combined_weights.loc[test_prices.index]
        test_meta = combined_meta.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices.loc[test_weights.index], test_weights, best_cfg.name, rf=rf, tx_cost=tx_cost)
        stitched_weights.append(test_weights)
        stitched_meta.append(test_meta)
        rows.append(
            {
                "window": window_id,
                "picked": best_cfg.name,
                "test_start": test_prices.index[0],
                "test_end": test_prices.index[-1],
                "cagr": float(test_metrics["cagr"]),
                "sharpe": float(test_metrics["sharpe"]),
                "mdd": float(test_metrics["mdd"]),
                "turnover": float(test_metrics["turnover"]),
            }
        )

    weights = pd.concat(stitched_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]
    meta_weights = pd.concat(stitched_meta).sort_index()
    meta_weights = meta_weights[~meta_weights.index.duplicated(keep="last")]

    metrics = performance_metrics(prices.loc[weights.index], weights, "Sleeve Sparse WFO Stitched OOS", rf=rf, tx_cost=tx_cost)
    metrics["avg_active_sleeves"] = active_tactical_sleeves(meta_weights)
    return {
        "windows": rows,
        "returns": metrics["returns"],  # type: ignore[index]
        "metrics": metrics,
        "meta_weights": meta_weights,
    }


def backtest(
    prices: pd.DataFrame,
    params: Optional[Dict[str, object]] = None,
    label: str = "Sleeve Sparse",
) -> Dict[str, object]:
    config = sleeve_sparse_candidates()[0]
    if params:
        config = SleeveSparseConfig(
            name=str(params.get("name", config.name)),
            base_meta=str(params.get("base_meta", config.base_meta)),
            tactical_top_k=int(params.get("tactical_top_k", config.tactical_top_k)),
            selection_frequency=str(params.get("selection_frequency", config.selection_frequency)),
            rank_buffer=int(params.get("rank_buffer", config.rank_buffer)),
            min_tactical_weight=float(params.get("min_tactical_weight", config.min_tactical_weight)),
            trade_band=float(params.get("trade_band", config.trade_band)),
            trade_mix=float(params.get("trade_mix", config.trade_mix)),
        )

    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    weights, meta_weights = run_sparse_sleeve_strategy(prices, macro, config, overlay=None, tx_cost=DEFAULT_TX, return_meta=True)
    result = performance_metrics(prices.loc[weights.index], weights, label, rf=DEFAULT_RF, tx_cost=DEFAULT_TX)
    result["avg_active_sleeves"] = active_tactical_sleeves(meta_weights)
    return result


def print_windows(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"\n{title}:")
    for row in rows:
        print(
            f"  W{row['window']:>2}: {row['test_start']:%Y-%m}→{row['test_end']:%Y-%m} "
            f"{row['picked']:<18} CAGR {row['cagr']:>6.1%} "
            f"Sharpe {row['sharpe']:>5.2f} Turn {row['turnover']:>6.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse sleeve meta research.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--llm-override-file")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource().fetch(args.start, end=args.end, refresh=args.refresh_cache)
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"), refresh=args.refresh_cache)
    overlay = load_llm_overlay(args.llm_override_file, prices.index)

    candidates = sleeve_sparse_candidates()
    recommended = candidates[0]

    full_rows: List[Dict[str, object]] = []
    meta_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cfg in candidates:
        weights, meta_weights = run_sparse_sleeve_strategy(prices, macro, cfg, overlay=overlay, tx_cost=tx_cost, return_meta=True)
        meta_cache[cfg.name] = {"asset": weights, "meta": meta_weights}
        row = performance_metrics(prices.loc[weights.index], weights, cfg.name, rf=args.rf, tx_cost=tx_cost)
        row["avg_active_sleeves"] = active_tactical_sleeves(meta_weights)
        full_rows.append(row)

    v12_cfg = resolve_meta_config("meta_63126_smooth")
    v12_weights = run_meta_strategy(prices, macro, v12_cfg, overlay=overlay, tx_cost=tx_cost)
    v12_result = performance_metrics(prices.loc[v12_weights.index], v12_weights, v12_cfg.name, rf=args.rf, tx_cost=tx_cost)
    v13_cfg = v13_candidates()[0]
    v13_wfo_fixed = fixed_sparse_walk_forward(prices, macro, v13_cfg, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    v9_wfo = parametric_v9_wfo(prices, [BASE_V9], rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=756, test_days=126)

    composite = fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost)
    eqwt_all7 = fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost)
    eqwt_risky = performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost)
    nifty = performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost)

    print_results_table(
        "FULL SAMPLE (1-day lag, 30 bps, all INR)",
        [*full_rows, v12_result, composite, eqwt_all7, eqwt_risky, nifty],
    )

    fixed_wfo = fixed_sleeve_sparse_walk_forward(prices, macro, recommended, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    adaptive_wfo = sleeve_sparse_walk_forward(prices, macro, candidates, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    v12_wfo = meta_walk_forward(prices, macro, meta_candidates(), rf=args.rf, tx_cost=tx_cost, overlay=overlay)

    oos_dates = fixed_wfo["metrics"]["returns"].index  # type: ignore[index]
    oos_rows: List[Dict[str, object]] = [fixed_wfo["metrics"], adaptive_wfo["metrics"], v12_wfo["metrics"], v13_wfo_fixed["metrics"], v9_wfo["metrics"]]  # type: ignore[list-item]
    oos_rows.append(fixed_weight_frame(prices.loc[oos_dates], COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost))
    oos_rows.append(fixed_weight_frame(prices.loc[oos_dates], {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost))
    oos_rows.extend(benchmark_oos_metrics(prices, oos_dates, rf=args.rf, tx_cost=tx_cost))
    print_results_table("WALK-FORWARD OOS (3Y train / 6M test, all INR)", oos_rows)

    print_windows("Fixed sleeve-sparse windows", fixed_wfo["windows"])  # type: ignore[arg-type]
    print_windows("Adaptive sleeve-sparse windows", adaptive_wfo["windows"])  # type: ignore[arg-type]

    latest = meta_cache[recommended.name]["asset"].iloc[-1]
    latest_meta = meta_cache[recommended.name]["meta"].iloc[-1]
    print(f"\nRecommended sleeve-sparse branch: {recommended.name}")
    print(f"Avg active tactical sleeves: {active_tactical_sleeves(meta_cache[recommended.name]['meta']):.2f}")
    print("Latest sleeve mix:")
    for sleeve, weight in latest_meta[latest_meta > 0.001].sort_values(ascending=False).items():
        print(f"  {sleeve:<10} {weight:>6.1%}")
    print("Latest asset allocation:")
    for asset, weight in latest[latest > 0.001].sort_values(ascending=False).items():
        print(f"  {asset:<10} {weight:>6.1%}")

    summary = {
        "full_sample": {
            row["label"]: {
                "cagr": float(row["cagr"]),
                "vol": float(row["vol"]),
                "sharpe": float(row["sharpe"]),
                "mdd": float(row["mdd"]),
                "calmar": float(row["calmar"]),
                "turnover": float(row["turnover"]),
                "avg_cash": float(row["avg_cash"]),
                "avg_active_sleeves": float(row.get("avg_active_sleeves", np.nan)),
            }
            for row in [*full_rows, v12_result, composite, eqwt_all7, eqwt_risky, nifty]
        },
        "wfo": {
            "fixed_sparse_sleeves": {
                "cagr": float(fixed_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(fixed_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(fixed_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(fixed_wfo["metrics"]["turnover"]),  # type: ignore[index]
                "avg_active_sleeves": float(fixed_wfo["metrics"]["avg_active_sleeves"]),  # type: ignore[index]
            },
            "adaptive_sparse_sleeves": {
                "cagr": float(adaptive_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(adaptive_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(adaptive_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(adaptive_wfo["metrics"]["turnover"]),  # type: ignore[index]
                "avg_active_sleeves": float(adaptive_wfo["metrics"]["avg_active_sleeves"]),  # type: ignore[index]
            },
            "v12": {
                "cagr": float(v12_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(v12_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(v12_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(v12_wfo["metrics"]["turnover"]),  # type: ignore[index]
            },
            "v13_fixed": {
                "cagr": float(v13_wfo_fixed["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(v13_wfo_fixed["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(v13_wfo_fixed["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(v13_wfo_fixed["metrics"]["turnover"]),  # type: ignore[index]
            },
            "v9": {
                "cagr": float(v9_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(v9_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(v9_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(v9_wfo["metrics"]["turnover"]),  # type: ignore[index]
            },
        },
    }
    RESULTS_CACHE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nResults cache written to:", RESULTS_CACHE)


if __name__ == "__main__":
    main()
