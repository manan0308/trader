#!/usr/bin/env python3
"""
ALPHA ENGINE v20 - DEEP BLEND SEARCH
====================================

Research-only search harness:
- Precompute several causal strategy families.
- Build convex blends of v9, momentum, and pullback mean-reversion.
- Add an outer smoothing/banding layer to reduce churn.
- Select blends inside each walk-forward train window, then stitch the OOS
  test windows and recompute returns with official delivery costs.

Guardrail:
This script is not an optimizer to blindly promote the top line. It is a way to
learn whether a genuinely better family exists after turnover and drawdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
from research.alpha_v12_meta_ensemble import BASE_V9
from research.alpha_v18_agile_rotation import (
    BROAD_SYMBOL_MAP,
    DEFAULT_DHAN,
    DEFAULT_ETF,
    DEFAULT_OVERLAY,
    ETF_SYMBOL_MAP,
    agile_candidates,
    load_parquet_prices,
    strip_metrics,
)
from research.alpha_v18_agile_rotation import run_agile_rotation_strategy
from research.alpha_v19_momentum_reversion import candidates as reversion_candidates
from research.alpha_v19_momentum_reversion import run_momentum_reversion_strategy
from strategy.v9_engine import (
    ALL,
    CACHE_DIR,
    DEFAULT_RF,
    DEFAULT_TX,
    NarrativeOverlay,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    run_strategy,
)


DEFAULT_OUTPUT = CACHE_DIR / "alpha_v20_deep_blend_search_results.json"


@dataclass(frozen=True)
class BlendSpec:
    name: str
    weights: Dict[str, float]
    band: float
    step: float


def smooth_weights(raw: pd.DataFrame, band: float, step: float) -> pd.DataFrame:
    raw = raw.reindex(columns=ALL).fillna(0.0)
    if raw.empty:
        return raw
    out = pd.DataFrame(0.0, index=raw.index, columns=ALL)
    current = raw.iloc[0].copy()
    if current.sum() > 0:
        current = current / current.sum()
    else:
        current["CASH"] = 1.0
    out.iloc[0] = current

    for i, dt in enumerate(raw.index[1:], start=1):
        target = raw.loc[dt].copy()
        target = target.clip(lower=0.0)
        if target.sum() > 0:
            target = target / target.sum()
        else:
            target["CASH"] = 1.0
        proposal = current * (1.0 - step) + target * step
        proposal = proposal.clip(lower=0.0)
        proposal = proposal / proposal.sum() if proposal.sum() > 0 else target
        if float((proposal - current).abs().max()) > band:
            current = proposal
        out.iloc[i] = current
    return out


def blend_family(base_weights: Dict[str, pd.DataFrame], spec: BlendSpec) -> pd.DataFrame:
    blended = pd.DataFrame(0.0, index=next(iter(base_weights.values())).index, columns=ALL)
    for name, weight in spec.weights.items():
        if weight <= 0.0:
            continue
        blended = blended.add(base_weights[name].reindex(columns=ALL).fillna(0.0) * weight, fill_value=0.0)
    row_sum = blended.sum(axis=1).replace(0.0, np.nan)
    blended = blended.div(row_sum, axis=0).fillna(0.0)
    return smooth_weights(blended, band=spec.band, step=spec.step)


def generate_blend_specs(base_names: Iterable[str], quick: bool = False) -> List[BlendSpec]:
    names = set(base_names)
    specs: List[BlendSpec] = []
    bands = [0.00, 0.05] if quick else [0.00, 0.025, 0.05, 0.075]
    steps = [0.75, 1.00] if quick else [0.50, 0.75, 1.00]

    def add(name: str, weights: Dict[str, float]) -> None:
        for band in bands:
            for step in steps:
                if band == 0.0 and step != 1.0:
                    continue
                suffix = f"b{int(band * 1000):03d}_s{int(step * 100):03d}"
                specs.append(BlendSpec(name=f"{name}_{suffix}", weights=weights, band=band, step=step))

    for base in sorted(names):
        add(base, {base: 1.0})

    momentum_names = [name for name in names if name.startswith("v18_")]
    reversion_names = [name for name in names if name.startswith("v19_")]
    for mom in momentum_names:
        for v9_weight in ([0.80, 0.70, 0.50] if quick else [0.90, 0.80, 0.70, 0.60, 0.50]):
            add(f"blend_{int(v9_weight*100)}v9_{int((1-v9_weight)*100)}{mom}", {"v9": v9_weight, mom: 1.0 - v9_weight})

    for rev in reversion_names:
        for v9_weight in ([0.80, 0.70, 0.50] if quick else [0.90, 0.80, 0.70, 0.60, 0.50]):
            add(f"blend_{int(v9_weight*100)}v9_{int((1-v9_weight)*100)}{rev}", {"v9": v9_weight, rev: 1.0 - v9_weight})

    for mom in momentum_names:
        for rev in reversion_names:
            triples = [
                (0.70, 0.20, 0.10),
                (0.70, 0.15, 0.15),
                (0.60, 0.25, 0.15),
                (0.60, 0.20, 0.20),
                (0.50, 0.35, 0.15),
                (0.50, 0.25, 0.25),
            ]
            if quick:
                triples = [(0.70, 0.20, 0.10), (0.60, 0.25, 0.15), (0.50, 0.25, 0.25)]
            for v9_weight, mom_weight, rev_weight in triples:
                add(
                    f"blend_{int(v9_weight*100)}v9_{int(mom_weight*100)}{mom}_{int(rev_weight*100)}{rev}",
                    {"v9": v9_weight, mom: mom_weight, rev: rev_weight},
                )
    return specs


def score(metrics: Dict[str, object]) -> float:
    sharpe = float(metrics["sharpe"])
    calmar = float(metrics["calmar"])
    cagr = float(metrics["cagr"])
    turnover = float(metrics["turnover"])
    drawdown = abs(float(metrics["mdd"]))
    return sharpe + 0.35 * calmar + 0.35 * cagr - 0.06 * turnover - 0.20 * drawdown


def stitched_wfo(
    prices: pd.DataFrame,
    candidates_by_name: Dict[str, pd.DataFrame],
    rf: float,
    tx_cost: float,
    train_days: int,
    test_days: int,
    cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
    max_candidates: int,
) -> Dict[str, object]:
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    picked_rows: List[Dict[str, object]] = []
    stitched_weights: List[pd.DataFrame] = []
    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        train_prices = prices.iloc[start_i:mid_i]
        test_prices = prices.iloc[mid_i:end_i]
        scored = []
        for label, weights in candidates_by_name.items():
            train_weights = weights.loc[train_prices.index]
            metrics = performance_metrics(
                train_prices,
                train_weights,
                label,
                rf=rf,
                tx_cost=tx_cost,
                cost_model=cost_model,
                base_value=base_value,
            )
            scored.append((score(metrics), label, metrics))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_label, best_train = scored[0]
        test_weights = candidates_by_name[best_label].loc[test_prices.index]
        test_metrics = performance_metrics(
            test_prices,
            test_weights,
            best_label,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )
        picked_rows.append(
            {
                "window": window_id,
                "test_start": test_prices.index[0].strftime("%Y-%m-%d"),
                "test_end": test_prices.index[-1].strftime("%Y-%m-%d"),
                "picked": best_label,
                "train_score": float(best_score),
                "train_cagr": float(best_train["cagr"]),
                "train_sharpe": float(best_train["sharpe"]),
                "train_mdd": float(best_train["mdd"]),
                "train_turnover": float(best_train["turnover"]),
                "test_cagr": float(test_metrics["cagr"]),
                "test_sharpe": float(test_metrics["sharpe"]),
                "test_mdd": float(test_metrics["mdd"]),
            }
        )
        stitched_weights.append(test_weights)

    if not stitched_weights:
        return {"metrics": {}, "windows": [], "candidate_count": len(candidates_by_name)}

    weights = pd.concat(stitched_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]
    metrics = performance_metrics(
        prices.loc[weights.index],
        weights,
        f"v20_train_selected_top{max_candidates}",
        rf=rf,
        tx_cost=tx_cost,
        cost_model=cost_model,
        base_value=base_value,
    )
    return {
        "metrics": metrics,
        "windows": picked_rows,
        "candidate_count": len(candidates_by_name),
    }


def precompute_base_weights(prices: pd.DataFrame, overlay: Optional[NarrativeOverlay] = None) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {
        "v9": run_strategy(prices, BASE_V9, overlay=overlay).reindex(columns=ALL).fillna(0.0)
    }
    for cfg in agile_candidates():
        if cfg.name in {"v18_aggressive_rotation", "v18_momentum_heavy_rotation", "v18_balanced_rotation"}:
            out[cfg.name] = run_agile_rotation_strategy(prices, cfg, overlay=overlay).reindex(columns=ALL).fillna(0.0)
    for cfg in reversion_candidates():
        out[cfg.name] = run_momentum_reversion_strategy(prices, cfg, overlay=overlay).reindex(columns=ALL).fillna(0.0)
    return out


def run_dataset(
    name: str,
    prices: pd.DataFrame,
    overlay: Optional[NarrativeOverlay],
    rf: float,
    tx_cost: float,
    train_days: int,
    test_days: int,
    cost_model: Optional[IndianDeliveryCostModel],
    preselect_cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
    top_n: int,
    quick: bool,
) -> Dict[str, object]:
    base_weights = precompute_base_weights(prices, overlay=None)
    specs = generate_blend_specs(base_weights.keys(), quick=quick)

    scored_full = []
    all_candidates: Dict[str, pd.DataFrame] = {}
    for spec in specs:
        weights = blend_family(base_weights, spec)
        metrics = performance_metrics(
            prices,
            weights,
            spec.name,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=preselect_cost_model,
            base_value=base_value,
        )
        scored_full.append((score(metrics), spec, metrics, weights))
    scored_full.sort(key=lambda item: item[0], reverse=True)

    selected_specs = scored_full[:top_n]
    for _, spec, _, weights in selected_specs:
        all_candidates[spec.name] = weights

    # Always keep known baselines even if the full-sample score would not rank
    # them in the top set.
    for label, weights in base_weights.items():
        all_candidates[label] = weights

    wfo = stitched_wfo(
        prices,
        all_candidates,
        rf=rf,
        tx_cost=tx_cost,
        train_days=train_days,
        test_days=test_days,
        cost_model=cost_model,
        base_value=base_value,
        max_candidates=top_n,
    )

    key_rows = []
    for label in [
        "v9",
        "v18_momentum_heavy_rotation",
        "v19_balanced_pullback",
        "v19_fast_pullback",
    ]:
        metrics = performance_metrics(
            prices,
            base_weights[label],
            label,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )
        key_rows.append(metrics)
    key_rows.append(wfo["metrics"])
    print_results_table(f"FULL/WFO KEY RESULTS - {name}", key_rows)

    return {
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "rows": int(len(prices)),
        "generated_specs": len(specs),
        "candidate_count_used_in_wfo": len(all_candidates),
        "preselect_cost_model": "india_delivery" if preselect_cost_model is not None else "flat",
        "top_preselect_full_sample": [
            {
                "rank": rank,
                "score": float(item_score),
                "spec": asdict(spec),
                "preselect_metrics": strip_metrics(metrics),
                "official_metrics": strip_metrics(
                    performance_metrics(
                        prices,
                        weights,
                        spec.name,
                        rf=rf,
                        tx_cost=tx_cost,
                        cost_model=cost_model,
                        base_value=base_value,
                    )
                ),
            }
            for rank, (item_score, spec, metrics, weights) in enumerate(selected_specs[:25], start=1)
        ],
        "walk_forward": {
            "metrics": strip_metrics(wfo["metrics"]),
            "windows": wfo["windows"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep blend search over v9/momentum/reversion families.")
    parser.add_argument("--dhan-parquet", default=str(DEFAULT_DHAN))
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--overlay-file", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--cost-model", choices=["flat", "india_delivery"], default="india_delivery")
    parser.add_argument("--preselect-cost-model", choices=["flat", "india_delivery"], default="flat")
    parser.add_argument("--base-value", type=float, default=1_000_000.0)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--quick", action="store_true", help="Use a smaller blend grid for interactive research.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000.0
    cost_model = resolve_cost_model(args.cost_model)
    preselect_cost_model = resolve_cost_model(args.preselect_cost_model)
    output_path = Path(args.output).expanduser().resolve()
    overlay_path = Path(args.overlay_file).expanduser().resolve() if args.overlay_file else None

    dhan_prices = load_parquet_prices(Path(args.dhan_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    overlay = load_llm_overlay(str(overlay_path), dhan_prices.index) if overlay_path and overlay_path.exists() else None

    datasets: Dict[str, object] = {
        "dhan_fy14_fy26": run_dataset(
            "Dhan FY14-FY26",
            dhan_prices,
            overlay,
            rf=args.rf,
            tx_cost=tx_cost,
            train_days=756,
            test_days=126,
            cost_model=cost_model,
            preselect_cost_model=preselect_cost_model,
            base_value=args.base_value,
            top_n=args.top_n,
            quick=args.quick,
        )
    }

    etf_path = Path(args.etf_parquet).expanduser().resolve()
    if etf_path.exists():
        etf_prices = load_parquet_prices(etf_path, ETF_SYMBOL_MAP)
        etf_overlay = load_llm_overlay(str(overlay_path), etf_prices.index) if overlay_path and overlay_path.exists() else None
        datasets["etf_fy24_fy26"] = run_dataset(
            "Dhan ETF FY24-FY26",
            etf_prices,
            etf_overlay,
            rf=args.rf,
            tx_cost=tx_cost,
            train_days=252,
            test_days=63,
            cost_model=cost_model,
            preselect_cost_model=preselect_cost_model,
            base_value=args.base_value,
            top_n=args.top_n,
            quick=args.quick,
        )

    payload = {
        "tx_cost": tx_cost,
        "cost_model": args.cost_model,
        "preselect_cost_model": args.preselect_cost_model,
        "base_value": args.base_value,
        "rf": args.rf,
        "top_n": args.top_n,
        "quick": args.quick,
        "datasets": datasets,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
