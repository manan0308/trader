#!/usr/bin/env python3
"""
ALPHA ENGINE v13 — SPARSE META ENSEMBLE
=======================================

Research goal:
- Keep the slow, honest v12 architecture.
- Allow zero weights in many assets through a final sparse selection layer.
- Compare concentrated versions fairly against broad v12, v9, and passive benchmarks.

Design:
1. Run the broad v12 meta allocator to produce baseline asset weights.
2. Preserve the broad model's cash level and total risky budget.
3. Concentrate the risky sleeve into the top-k assets by broad weight.
4. Offer slower weekly/monthly selection variants to control turnover.
5. Keep the LLM as a governor only: sparse selection happens after overlay-aware broad weights.
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
    RISKY,
    NarrativeOverlay,
    YahooFinanceSource,
    benchmark_oos_metrics,
    benchmark_weights,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    schedule_flags,
)
from research.alpha_v10_canary_research import COMPOSITE_FIXED_WEIGHTS
from research.alpha_v11_macro_value_research import fetch_macro_panel, parametric_v9_wfo
from research.alpha_v12_meta_ensemble import (
    BASE_V9,
    MetaConfig,
    fixed_weight_frame,
    meta_candidates,
    meta_walk_forward,
    run_meta_strategy,
)


RESULTS_CACHE = CACHE_DIR / "alpha_v13_sparse_meta_results.json"


@dataclass(frozen=True)
class SparseMetaConfig:
    name: str
    base_meta: str
    top_k: int
    selection_frequency: str
    rank_buffer: int
    min_keep_weight: float
    preserve_risky_budget: bool
    trade_band: float


def active_positions(weights: pd.DataFrame) -> float:
    return float((weights[RISKY] > 0.001).sum(axis=1).mean())


def resolve_meta_config(name: str) -> MetaConfig:
    mapping = {cfg.name: cfg for cfg in meta_candidates()}
    if name not in mapping:
        raise KeyError(f"Unknown base meta config: {name}")
    return mapping[name]


def sparse_candidates() -> List[SparseMetaConfig]:
    return [
        SparseMetaConfig(
            name="sparse_top4_monthly_slow",
            base_meta="meta_126252_direct",
            top_k=4,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=True,
            trade_band=0.04,
        ),
        SparseMetaConfig(
            name="sparse_top3_monthly_slow",
            base_meta="meta_126252_direct",
            top_k=3,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=True,
            trade_band=0.04,
        ),
        SparseMetaConfig(
            name="sparse_top4_monthly_smooth",
            base_meta="meta_63126_smooth",
            top_k=4,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=True,
            trade_band=0.04,
        ),
        SparseMetaConfig(
            name="sparse_top3_monthly_smooth",
            base_meta="meta_63126_smooth",
            top_k=3,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=True,
            trade_band=0.04,
        ),
        SparseMetaConfig(
            name="sparse_top4_monthly_cashspill",
            base_meta="meta_126252_direct",
            top_k=4,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=False,
            trade_band=0.04,
        ),
        SparseMetaConfig(
            name="sparse_top3_monthly_cashspill",
            base_meta="meta_126252_direct",
            top_k=3,
            selection_frequency="MONTHLY",
            rank_buffer=1,
            min_keep_weight=0.04,
            preserve_risky_budget=False,
            trade_band=0.04,
        ),
    ]


def risk_preserving_candidates() -> List[SparseMetaConfig]:
    return [cfg for cfg in sparse_candidates() if cfg.preserve_risky_budget]


def cash_spill_candidates() -> List[SparseMetaConfig]:
    return [cfg for cfg in sparse_candidates() if not cfg.preserve_risky_budget]


def selection_schedule(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    if frequency == "DAILY":
        return pd.Series(True, index=index)
    return schedule_flags(index, frequency)


def pick_assets(
    risky_row: pd.Series,
    previous: List[str],
    config: SparseMetaConfig,
) -> List[str]:
    ranked = risky_row.sort_values(ascending=False)
    ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)

    keep: List[str] = []
    for asset in previous:
        if asset not in risky_row.index:
            continue
        if int(ranks[asset]) <= config.top_k + config.rank_buffer or float(risky_row[asset]) >= config.min_keep_weight:
            keep.append(asset)

    picks = list(keep)
    for asset in ranked.index:
        if asset in picks:
            continue
        if float(risky_row[asset]) <= 0.001:
            continue
        picks.append(asset)
        if len(picks) >= config.top_k:
            break

    if not picks:
        picks = ranked.head(config.top_k).index.tolist()
    return picks[: config.top_k]


def build_sparse_target(row: pd.Series, selected: List[str], config: SparseMetaConfig) -> pd.Series:
    target = pd.Series(0.0, index=ALL, dtype=float)
    risky_row = row[RISKY].clip(lower=0.0)

    if config.preserve_risky_budget:
        risky_budget = float(risky_row.sum())
        cash_weight = float(row["CASH"])
    else:
        risky_budget = float(risky_row[selected].sum()) if selected else 0.0
        cash_weight = max(0.0, 1.0 - risky_budget)

    if selected:
        kept = risky_row[selected]
        if float(kept.sum()) > 0:
            target[selected] = kept / float(kept.sum()) * risky_budget
        else:
            target[selected] = risky_budget / len(selected)

    target["CASH"] = cash_weight
    if float(target.sum()) <= 0:
        target["CASH"] = 1.0
    return target / float(target.sum())


def sparsify_weights(broad_weights: pd.DataFrame, config: SparseMetaConfig) -> pd.DataFrame:
    schedule = selection_schedule(broad_weights.index, config.selection_frequency)
    sparse = pd.DataFrame(0.0, index=broad_weights.index, columns=ALL)

    current = pd.Series(0.0, index=ALL, dtype=float)
    current["CASH"] = 1.0
    selected: List[str] = []

    for dt in broad_weights.index:
        row = broad_weights.loc[dt]
        if bool(schedule.loc[dt]) or not selected:
            selected = pick_assets(row[RISKY], selected, config)
            target = build_sparse_target(row, selected, config)
            if float((target - current).abs().max()) > config.trade_band:
                current = target
        sparse.loc[dt] = current

    sparse = sparse.div(sparse.sum(axis=1), axis=0).fillna(0.0)
    return sparse


def run_sparse_meta_strategy(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: SparseMetaConfig,
    overlay: Optional[NarrativeOverlay] = None,
    tx_cost: float = DEFAULT_TX,
) -> pd.DataFrame:
    base_cfg = resolve_meta_config(config.base_meta)
    broad = run_meta_strategy(prices, macro, base_cfg, overlay=overlay, tx_cost=tx_cost)
    return sparsify_weights(broad, config)


def sparse_strategy_score(metrics: Dict[str, object]) -> float:
    sharpe = float(metrics["sharpe"])
    calmar = float(metrics["calmar"])
    turnover = float(metrics["turnover"])
    return sharpe + 0.30 * calmar - 0.08 * turnover


def sparse_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    candidates: List[SparseMetaConfig],
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

    if not windows:
        empty = pd.Series(dtype=float)
        return {
            "windows": [],
            "returns": empty,
            "metrics": {
                "label": "Sparse Meta WFO Stitched OOS",
                "weights": pd.DataFrame(columns=ALL, dtype=float),
                "returns": empty,
                "equity": empty,
                "cagr": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "mdd": np.nan,
                "calmar": np.nan,
                "turnover": np.nan,
                "avg_cash": np.nan,
                "avg_active": np.nan,
            },
        }

    stitched_returns: List[pd.Series] = []
    stitched_weights: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        train_prices = prices.iloc[start_i:mid_i]
        train_macro = macro.reindex(train_prices.index).ffill()
        test_prices = prices.iloc[mid_i:end_i]
        train_overlay = None
        if overlay is not None:
            train_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[train_prices.index],
                asset_bias=overlay.asset_bias.loc[train_prices.index],
            )

        best_cfg = candidates[0]
        best_score = -np.inf
        train_broad_cache: Dict[str, pd.DataFrame] = {}
        for cfg in candidates:
            broad = train_broad_cache.get(cfg.base_meta)
            if broad is None:
                broad = run_meta_strategy(
                    train_prices,
                    train_macro,
                    resolve_meta_config(cfg.base_meta),
                    overlay=train_overlay,
                    tx_cost=tx_cost,
                )
                train_broad_cache[cfg.base_meta] = broad
            train_weights = sparsify_weights(broad, cfg)
            train_metrics = performance_metrics(train_prices.loc[train_weights.index], train_weights, cfg.name, rf=rf, tx_cost=tx_cost)
            score = sparse_strategy_score(train_metrics)
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
        combined_broad = run_meta_strategy(
            combined_prices,
            combined_macro,
            resolve_meta_config(best_cfg.base_meta),
            overlay=combined_overlay,
            tx_cost=tx_cost,
        )
        combined_weights = sparsify_weights(combined_broad, best_cfg)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices.loc[test_weights.index], test_weights, best_cfg.name, rf=rf, tx_cost=tx_cost)

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
        "windows": rows,
        "returns": stitched,
        "metrics": {
            "label": "Sparse Meta WFO Stitched OOS",
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
            "avg_active": active_positions(weights),
        },
    }


def fixed_sparse_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: SparseMetaConfig,
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
        combined_broad = run_meta_strategy(
            combined_prices,
            combined_macro,
            resolve_meta_config(config.base_meta),
            overlay=combined_overlay,
            tx_cost=tx_cost,
        )
        combined_weights = sparsify_weights(combined_broad, config)
        test_prices = prices.iloc[mid_i:end_i]
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices.loc[test_weights.index], test_weights, config.name, rf=rf, tx_cost=tx_cost)
        stitched_weights.append(test_weights)
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
    metrics = performance_metrics(prices.loc[weights.index], weights, f"{config.name} WFO Fixed OOS", rf=rf, tx_cost=tx_cost)
    metrics["avg_active"] = active_positions(weights)
    return {
        "windows": rows,
        "returns": metrics["returns"],  # type: ignore[index]
        "metrics": metrics,
    }


def add_active_info(metrics: Dict[str, object]) -> Dict[str, object]:
    metrics = dict(metrics)
    metrics["avg_active"] = active_positions(metrics["weights"])  # type: ignore[index]
    return metrics


def backtest(
    prices: pd.DataFrame,
    params: Optional[Dict[str, object]] = None,
    label: str = "Sparse Meta",
) -> Dict[str, object]:
    config = sparse_candidates()[0]
    if params:
        config = SparseMetaConfig(
            name=str(params.get("name", config.name)),
            base_meta=str(params.get("base_meta", config.base_meta)),
            top_k=int(params.get("top_k", config.top_k)),
            selection_frequency=str(params.get("selection_frequency", config.selection_frequency)),
            rank_buffer=int(params.get("rank_buffer", config.rank_buffer)),
            min_keep_weight=float(params.get("min_keep_weight", config.min_keep_weight)),
            preserve_risky_budget=bool(params.get("preserve_risky_budget", config.preserve_risky_budget)),
            trade_band=float(params.get("trade_band", config.trade_band)),
        )

    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    weights = run_sparse_meta_strategy(prices, macro, config, overlay=None, tx_cost=DEFAULT_TX)
    result = performance_metrics(prices.loc[weights.index], weights, label, rf=DEFAULT_RF, tx_cost=DEFAULT_TX)
    result["avg_active"] = active_positions(weights)
    return result


def print_sparse_windows(rows: List[Dict[str, object]]) -> None:
    print("\nSparse WFO picks:")
    for row in rows:
        print(
            f"  W{row['window']:>2}: {row['test_start']:%Y-%m}→{row['test_end']:%Y-%m} "
            f"{row['picked']:<22} CAGR {row['cagr']:>6.1%} "
            f"Sharpe {row['sharpe']:>5.2f} Turn {row['turnover']:>6.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse meta-ensemble tactical allocation research.")
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

    candidates = sparse_candidates()
    preserving = risk_preserving_candidates()
    cashspill = cash_spill_candidates()
    recommended = preserving[0]

    full_rows: List[Dict[str, object]] = []
    sparse_outputs: Dict[str, pd.DataFrame] = {}
    broad_cache: Dict[str, pd.DataFrame] = {}
    for cfg in candidates:
        broad = broad_cache.get(cfg.base_meta)
        if broad is None:
            broad = run_meta_strategy(prices, macro, resolve_meta_config(cfg.base_meta), overlay=overlay, tx_cost=tx_cost)
            broad_cache[cfg.base_meta] = broad
        weights = sparsify_weights(broad, cfg)
        sparse_outputs[cfg.name] = weights
        row = performance_metrics(prices.loc[weights.index], weights, cfg.name, rf=args.rf, tx_cost=tx_cost)
        row["avg_active"] = active_positions(weights)
        full_rows.append(row)

    v12_cfg = resolve_meta_config("meta_63126_smooth")
    v12_weights = run_meta_strategy(prices, macro, v12_cfg, overlay=overlay, tx_cost=tx_cost)
    v12_result = performance_metrics(prices.loc[v12_weights.index], v12_weights, v12_cfg.name, rf=args.rf, tx_cost=tx_cost)
    v12_result["avg_active"] = active_positions(v12_weights)

    composite = fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost)
    eqwt_all7 = fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost)
    eqwt_risky = performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost)
    nifty = performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost)

    print_results_table(
        "FULL SAMPLE (1-day lag, 30 bps, all INR)",
        [*full_rows, v12_result, composite, eqwt_all7, eqwt_risky, nifty],
    )

    sparse_wfo = sparse_walk_forward(prices, macro, preserving, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    fixed_sparse_wfo = fixed_sparse_walk_forward(prices, macro, recommended, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    cashspill_wfo = sparse_walk_forward(prices, macro, cashspill, rf=args.rf, tx_cost=tx_cost, overlay=overlay) if cashspill else None
    v12_wfo = meta_walk_forward(prices, macro, meta_candidates(), rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    v9_wfo = parametric_v9_wfo(prices, [BASE_V9], rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=756, test_days=126)

    oos_dates = fixed_sparse_wfo["metrics"]["returns"].index  # type: ignore[index]
    oos_rows: List[Dict[str, object]] = [fixed_sparse_wfo["metrics"], sparse_wfo["metrics"], v12_wfo["metrics"], v9_wfo["metrics"]]  # type: ignore[list-item]
    oos_rows.append(fixed_weight_frame(prices.loc[oos_dates], COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost))
    oos_rows.append(fixed_weight_frame(prices.loc[oos_dates], {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost))
    oos_rows.extend(benchmark_oos_metrics(prices, oos_dates, rf=args.rf, tx_cost=tx_cost))

    print_results_table("WALK-FORWARD OOS (3Y train / 6M test, all INR)", oos_rows)
    print("\nFixed sparse branch windows:")
    print_sparse_windows(fixed_sparse_wfo["windows"])  # type: ignore[arg-type]
    print_sparse_windows(sparse_wfo["windows"])  # type: ignore[arg-type]
    if cashspill_wfo is not None:
        metrics = cashspill_wfo["metrics"]  # type: ignore[index]
        print(
            "\nCash-spill reference branch:",
            f"CAGR {float(metrics['cagr']):.1%} |",
            f"Sharpe {float(metrics['sharpe']):.2f} |",
            f"MaxDD {float(metrics['mdd']):.1%} |",
            f"Turnover {float(metrics['turnover']):.0%} |",
            f"AvgCash {float(metrics['avg_cash']):.0%}",
        )

    latest = sparse_outputs[recommended.name].iloc[-1]
    print(f"\nRecommended sparse branch: {recommended.name}")
    print(f"Avg active risky positions: {active_positions(sparse_outputs[recommended.name]):.2f}")
    print("Latest allocation:")
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
                "avg_active": float(row.get("avg_active", np.nan)),
            }
            for row in [*full_rows, v12_result, composite, eqwt_all7, eqwt_risky, nifty]
        },
        "wfo": {
            "sparse": {
                "fixed_cagr": float(fixed_sparse_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "fixed_sharpe": float(fixed_sparse_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "fixed_mdd": float(fixed_sparse_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "fixed_turnover": float(fixed_sparse_wfo["metrics"]["turnover"]),  # type: ignore[index]
                "fixed_avg_active": float(fixed_sparse_wfo["metrics"]["avg_active"]),  # type: ignore[index]
                "cagr": float(sparse_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(sparse_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(sparse_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(sparse_wfo["metrics"]["turnover"]),  # type: ignore[index]
                "avg_active": float(sparse_wfo["metrics"]["avg_active"]),  # type: ignore[index]
            },
            "cashspill": (
                {
                    "cagr": float(cashspill_wfo["metrics"]["cagr"]),  # type: ignore[index]
                    "sharpe": float(cashspill_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                    "mdd": float(cashspill_wfo["metrics"]["mdd"]),  # type: ignore[index]
                    "turnover": float(cashspill_wfo["metrics"]["turnover"]),  # type: ignore[index]
                    "avg_active": float(cashspill_wfo["metrics"]["avg_active"]),  # type: ignore[index]
                }
                if cashspill_wfo is not None
                else {}
            ),
            "v12": {
                "cagr": float(v12_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(v12_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(v12_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(v12_wfo["metrics"]["turnover"]),  # type: ignore[index]
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
