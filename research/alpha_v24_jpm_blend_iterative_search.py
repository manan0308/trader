#!/usr/bin/env python3
"""
ALPHA ENGINE v24 - JPM BLEND ITERATIVE SEARCH
=============================================

Iterative search over blends of:
- v9 core
- JPM scorecard long-only momentum
- JPM diversified trend-following blend
- JPM slow absolute momentum

Intent:
- keep broad-history robustness as the anchor
- use the stronger paper-inspired challengers as return enhancers
- continuously search smooth convex blends rather than promoting a raw
  challenger directly
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
from research.alpha_v18_agile_rotation import (
    BROAD_SYMBOL_MAP,
    DEFAULT_DHAN,
    DEFAULT_ETF,
    ETF_SYMBOL_MAP,
    fixed_config_wfo,
    load_parquet_prices,
    rotation_summary,
)
from research.alpha_v23_jpm_momentum_playbook import (
    DEFAULT_RECENT,
    load_etf_prices,
    paper_variants,
    run_long_only_paper_variant,
)
from strategy.v9_engine import ALL, CACHE_DIR, DEFAULT_RF, RISKY, performance_metrics, run_strategy
from research.alpha_v12_meta_ensemble import BASE_V9


DEFAULT_BEST = CACHE_DIR / "alpha_v24_jpm_blend_best.json"
DEFAULT_HISTORY = CACHE_DIR / "alpha_v24_jpm_blend_history.jsonl"


@dataclass(frozen=True)
class BlendCandidate:
    weights: Dict[str, float]
    band: float
    step: float


def smooth_weights(raw: pd.DataFrame, band: float, step: float) -> pd.DataFrame:
    raw = raw.reindex(columns=ALL).fillna(0.0)
    if raw.empty:
        return raw
    out = pd.DataFrame(0.0, index=raw.index, columns=ALL, dtype=float)
    current = raw.iloc[0].copy()
    current = current / float(current.sum()) if float(current.sum()) > 0 else pd.Series([0, 0, 0, 0, 0, 0, 1], index=ALL, dtype=float)
    out.iloc[0] = current
    for i, dt in enumerate(raw.index[1:], start=1):
        target = raw.loc[dt].clip(lower=0.0)
        target = target / float(target.sum()) if float(target.sum()) > 0 else current
        proposal = current * (1.0 - step) + target * step
        proposal = proposal.clip(lower=0.0)
        proposal["CASH"] = max(0.0, 1.0 - float(proposal[RISKY].sum()))
        proposal = proposal / float(proposal.sum())
        if float((proposal - current).abs().max()) > band:
            current = proposal
        out.iloc[i] = current
    return out


def candidate_name(candidate: BlendCandidate) -> str:
    bits = []
    for key, value in sorted(candidate.weights.items()):
        bits.append(f"{int(round(value * 100)):02d}{key}")
    return f"v24_{'_'.join(bits)}_b{int(candidate.band*1000):03d}_s{int(candidate.step*100):03d}"


def normalize_weights(raw: Dict[str, float], component_names: List[str]) -> Dict[str, float]:
    weights = {name: max(0.0, float(raw.get(name, 0.0))) for name in component_names}
    weights["v9"] = max(weights.get("v9", 0.0), 0.35)
    active_challengers = sorted(
        [name for name in component_names if name != "v9" and weights.get(name, 0.0) > 0.0],
        key=lambda name: weights[name],
        reverse=True,
    )
    keep = {"v9", *active_challengers[:2]}
    trimmed = {name: weight for name, weight in weights.items() if name in keep and weight > 0.0}
    total = float(sum(trimmed.values()))
    if total <= 0.0:
        return {"v9": 1.0}
    return {name: weight / total for name, weight in trimmed.items()}


def sample_candidate(rng: random.Random, component_names: List[str]) -> BlendCandidate:
    challengers = [name for name in component_names if name != "v9"]
    chosen = ["v9"]
    extra_n = rng.choice([1, 2])
    chosen += rng.sample(challengers, k=extra_n)

    v9_weight = rng.choice([0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    remainder = max(0.0, 1.0 - v9_weight)
    if extra_n == 1:
        weights = {"v9": v9_weight, chosen[1]: remainder}
    else:
        split = rng.choice([0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75])
        weights = {
            "v9": v9_weight,
            chosen[1]: remainder * split,
            chosen[2]: remainder * (1.0 - split),
        }
    total = float(sum(weights.values()))
    weights = {k: float(v) / total for k, v in weights.items()}
    return BlendCandidate(
        weights=weights,
        band=rng.choice([0.00, 0.025, 0.05, 0.075]),
        step=rng.choice([0.50, 0.75, 1.00]),
    )


def row_to_candidate(row: Dict[str, object], component_names: List[str]) -> BlendCandidate:
    weights = normalize_weights(dict(row.get("weights", {})), component_names)
    band = float(row.get("band", 0.025))
    step = float(row.get("step", 1.0))
    return BlendCandidate(weights=weights, band=band, step=step)


def mutate_candidate(rng: random.Random, base: BlendCandidate, component_names: List[str]) -> BlendCandidate:
    challenger_names = [name for name in component_names if name != "v9"]
    weights = {name: float(base.weights.get(name, 0.0)) for name in component_names}

    if len([name for name in challenger_names if weights.get(name, 0.0) > 0.0]) < 2 and rng.random() < 0.4:
        extra = rng.choice([name for name in challenger_names if weights.get(name, 0.0) <= 0.0])
        weights[extra] = rng.choice([0.03, 0.05, 0.08, 0.10])

    for name in component_names:
        if name == "v9":
            weights[name] = max(0.0, weights.get(name, 0.0) + rng.uniform(-0.10, 0.10))
        else:
            weights[name] = max(0.0, weights.get(name, 0.0) + rng.uniform(-0.08, 0.08))
            if weights[name] < 0.025:
                weights[name] = 0.0

    weights = normalize_weights(weights, component_names)
    band_choices = [0.00, 0.025, 0.05, 0.075]
    step_choices = [0.50, 0.75, 1.00]
    band_idx = min(range(len(band_choices)), key=lambda i: abs(band_choices[i] - base.band))
    step_idx = min(range(len(step_choices)), key=lambda i: abs(step_choices[i] - base.step))
    band_idx = max(0, min(len(band_choices) - 1, band_idx + rng.choice([-1, 0, 0, 1])))
    step_idx = max(0, min(len(step_choices) - 1, step_idx + rng.choice([-1, 0, 0, 1])))
    return BlendCandidate(weights=weights, band=band_choices[band_idx], step=step_choices[step_idx])


def blend_family(base_weights: Dict[str, pd.DataFrame], candidate: BlendCandidate) -> pd.DataFrame:
    first = next(iter(base_weights.values()))
    blended = pd.DataFrame(0.0, index=first.index, columns=ALL, dtype=float)
    for name, weight in candidate.weights.items():
        blended = blended.add(base_weights[name].reindex(index=blended.index, columns=ALL).fillna(0.0) * weight, fill_value=0.0)
    row_sum = blended.sum(axis=1).replace(0.0, np.nan)
    blended = blended.div(row_sum, axis=0).fillna(0.0)
    return smooth_weights(blended, band=candidate.band, step=candidate.step)


def one_window_return(returns: pd.Series, start: str, end: str) -> float:
    if returns.empty:
        return float("nan")
    idx = returns.index
    start_dt = idx[idx <= pd.Timestamp(start)][-1]
    end_dt = idx[idx <= pd.Timestamp(end)][-1]
    window = returns.loc[start_dt:end_dt]
    return float((1.0 + window).prod() - 1.0)


def pack_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    return {
        "cagr": float(metrics["cagr"]),
        "sharpe": float(metrics["sharpe"]),
        "mdd": float(metrics["mdd"]),
        "turnover": float(metrics["turnover"]),
        "avg_cash": float(metrics["avg_cash"]),
    }


def evaluate_candidate(
    candidate: BlendCandidate,
    broad_prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    broad_components: Dict[str, pd.DataFrame],
    etf_components: Dict[str, pd.DataFrame],
    rf: float,
    tx_cost: float,
    cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
    train_days: int,
    test_days: int,
    baselines: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    broad_weights = blend_family(broad_components, candidate)
    etf_weights = blend_family(etf_components, candidate)
    label = candidate_name(candidate)

    broad_full = performance_metrics(
        broad_prices, broad_weights, f"{label}_broad", rf=rf, tx_cost=tx_cost, cost_model=cost_model, base_value=base_value
    )
    etf_full = performance_metrics(
        etf_prices, etf_weights, f"{label}_etf", rf=rf, tx_cost=tx_cost, cost_model=cost_model, base_value=base_value
    )

    broad_wfo = fixed_config_wfo(
        broad_prices,
        runner=lambda frame: broad_weights.loc[frame.index],
        label=f"{label}_broad_wfo",
        rf=rf,
        tx_cost=tx_cost,
        train_days=train_days,
        test_days=test_days,
        cost_model=cost_model,
        base_value=base_value,
    )
    etf_wfo = fixed_config_wfo(
        etf_prices,
        runner=lambda frame: etf_weights.loc[frame.index],
        label=f"{label}_etf_wfo",
        rf=rf,
        tx_cost=tx_cost,
        train_days=min(train_days, 504),
        test_days=min(test_days, 63),
        cost_model=cost_model,
        base_value=base_value,
    )

    april_return = one_window_return(pd.Series(etf_full["returns"]), "2026-04-01", "2026-04-17")
    dec_window = one_window_return(pd.Series(broad_full["returns"]), "2024-12-01", "2025-12-31")

    broad_full_clean = pack_metrics(broad_full)
    etf_full_clean = pack_metrics(etf_full)
    broad_wfo_clean = pack_metrics(broad_wfo["metrics"])
    etf_wfo_clean = pack_metrics(etf_wfo["metrics"])

    loss = 0.0
    loss -= 8.0 * (broad_wfo_clean["sharpe"] - baselines["broad_wfo"]["sharpe"])
    loss -= 4.0 * (broad_full_clean["sharpe"] - baselines["broad_full"]["sharpe"])
    loss -= 3.0 * (broad_full_clean["cagr"] - baselines["broad_full"]["cagr"])
    loss -= 3.0 * (etf_full_clean["sharpe"] - baselines["etf_full"]["sharpe"])
    loss -= 2.0 * (etf_full_clean["cagr"] - baselines["etf_full"]["cagr"])
    loss -= 2.0 * (etf_wfo_clean["sharpe"] - baselines["etf_wfo"]["sharpe"])
    loss -= 1.0 * (april_return - baselines["april_return"]["value"])
    loss -= 1.5 * (dec_window - baselines["dec_window"]["value"])

    loss += 8.0 * max(0.0, abs(broad_full_clean["mdd"]) - abs(baselines["broad_full"]["mdd"]) - 0.01)
    loss += 5.0 * max(0.0, abs(broad_wfo_clean["mdd"]) - abs(baselines["broad_wfo"]["mdd"]) - 0.01)
    loss += 2.0 * max(0.0, broad_full_clean["turnover"] - 1.75)
    loss += 1.5 * max(0.0, broad_wfo_clean["turnover"] - 1.75)
    loss += 1.0 * max(0.0, etf_full_clean["turnover"] - 2.25)
    loss += 1.0 * max(0.0, etf_wfo_clean["turnover"] - 2.25)

    return {
        "name": label,
        "loss": float(loss),
        "weights": candidate.weights,
        "band": float(candidate.band),
        "step": float(candidate.step),
        "broad_full": broad_full_clean,
        "broad_wfo": broad_wfo_clean,
        "etf_full": etf_full_clean,
        "etf_wfo": etf_wfo_clean,
        "april_2026_return": float(april_return),
        "dec24_dec25_return": float(dec_window),
        "broad_rotation": rotation_summary(broad_weights),
        "etf_rotation": rotation_summary(etf_weights),
    }


def save_best(best_rows: List[Dict[str, object]], payload_path: Path, history_path: Path, cycle: int, samples_done: int, baselines: Dict[str, Dict[str, float]]) -> None:
    payload = {
        "updated_at_epoch": time.time(),
        "cycle": cycle,
        "samples_done": samples_done,
        "baselines": baselines,
        "best": best_rows,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def precompute_components(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {"v9": run_strategy(prices, BASE_V9, overlay=None).reindex(columns=ALL).fillna(0.0)}
    variants = {cfg.name: cfg for cfg in paper_variants()}
    for key in ["v23_jpm_scorecard8_top3", "v23_jpm_dtf3signal_top3", "v23_jpm_abs12_allpos", "v23_jpm_hybrid6_top3"]:
        out[key] = run_long_only_paper_variant(prices, variants[key]).reindex(columns=ALL).fillna(0.0)
    return out


def load_previous_best(best_path: Path, component_names: List[str], top_k: int) -> List[Dict[str, object]]:
    if not best_path.exists():
        return []
    try:
        payload = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rows = payload.get("best", [])
    if not isinstance(rows, list):
        return []
    cleaned: List[Dict[str, object]] = []
    for row in rows[:top_k]:
        if not isinstance(row, dict):
            continue
        candidate = row_to_candidate(row, component_names)
        patched = dict(row)
        patched["weights"] = candidate.weights
        patched["band"] = candidate.band
        patched["step"] = candidate.step
        cleaned.append(patched)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative blend search around v9 and JPM-paper-inspired momentum variants.")
    parser.add_argument("--broad-parquet", default=str(DEFAULT_DHAN))
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--recent-csv", default=str(DEFAULT_RECENT))
    parser.add_argument("--best-json", default=str(DEFAULT_BEST))
    parser.add_argument("--history-jsonl", default=str(DEFAULT_HISTORY))
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--cost-model", choices=["flat", "india_delivery"], default="india_delivery")
    parser.add_argument("--base-value", type=float, default=1_000_000.0)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--samples-per-cycle", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tx_cost = args.tx_bps / 10_000.0
    cost_model = resolve_cost_model(args.cost_model)

    broad_prices = load_parquet_prices(Path(args.broad_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    etf_prices = load_etf_prices(Path(args.etf_parquet).expanduser().resolve(), Path(args.recent_csv).expanduser().resolve())

    broad_components = precompute_components(broad_prices)
    etf_components = precompute_components(etf_prices)

    base_broad_full = performance_metrics(broad_prices, broad_components["v9"], "v9_broad", rf=args.rf, tx_cost=tx_cost, cost_model=cost_model, base_value=args.base_value)
    base_etf_full = performance_metrics(etf_prices, etf_components["v9"], "v9_etf", rf=args.rf, tx_cost=tx_cost, cost_model=cost_model, base_value=args.base_value)
    base_broad_wfo = fixed_config_wfo(
        broad_prices,
        runner=lambda frame: broad_components["v9"].loc[frame.index],
        label="v9_broad_wfo",
        rf=args.rf,
        tx_cost=tx_cost,
        train_days=args.train_days,
        test_days=args.test_days,
        cost_model=cost_model,
        base_value=args.base_value,
    )
    base_etf_wfo = fixed_config_wfo(
        etf_prices,
        runner=lambda frame: etf_components["v9"].loc[frame.index],
        label="v9_etf_wfo",
        rf=args.rf,
        tx_cost=tx_cost,
        train_days=min(args.train_days, 504),
        test_days=min(args.test_days, 63),
        cost_model=cost_model,
        base_value=args.base_value,
    )
    baselines = {
        "broad_full": pack_metrics(base_broad_full),
        "etf_full": pack_metrics(base_etf_full),
        "broad_wfo": pack_metrics(base_broad_wfo["metrics"]),
        "etf_wfo": pack_metrics(base_etf_wfo["metrics"]),
        "april_return": {"value": one_window_return(pd.Series(base_etf_full["returns"]), "2026-04-01", "2026-04-17")},
        "dec_window": {"value": one_window_return(pd.Series(base_broad_full["returns"]), "2024-12-01", "2025-12-31")},
    }

    component_names = list(broad_components.keys())
    best_json_path = Path(args.best_json).expanduser().resolve()
    history_jsonl_path = Path(args.history_jsonl).expanduser().resolve()
    best_rows: List[Dict[str, object]] = load_previous_best(best_json_path, component_names, args.top_k)
    cycle = 0
    samples_done = 0

    while True:
        cycle += 1
        rows = list(best_rows)
        for _ in range(args.samples_per_cycle):
            if best_rows and rng.random() < 0.65:
                seed_row = rng.choice(best_rows[: min(5, len(best_rows))])
                candidate = mutate_candidate(rng, row_to_candidate(seed_row, component_names), component_names)
            else:
                candidate = sample_candidate(rng, component_names)
            rows.append(
                evaluate_candidate(
                    candidate,
                    broad_prices=broad_prices,
                    etf_prices=etf_prices,
                    broad_components=broad_components,
                    etf_components=etf_components,
                    rf=args.rf,
                    tx_cost=tx_cost,
                    cost_model=cost_model,
                    base_value=args.base_value,
                    train_days=args.train_days,
                    test_days=args.test_days,
                    baselines=baselines,
                )
            )
            samples_done += 1
        rows.sort(key=lambda item: float(item["loss"]))
        deduped: List[Dict[str, object]] = []
        seen = set()
        for row in rows:
            key = row["name"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= args.top_k:
                break
        best_rows = deduped
        save_best(best_rows, best_json_path, history_jsonl_path, cycle, samples_done, baselines)
        best = best_rows[0] if best_rows else {}
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "samples_done": samples_done,
                    "best_name": best.get("name"),
                    "best_loss": best.get("loss"),
                    "best_broad_wfo_sharpe": best.get("broad_wfo", {}).get("sharpe") if best else None,
                    "best_broad_full_cagr": best.get("broad_full", {}).get("cagr") if best else None,
                    "best_etf_full_cagr": best.get("etf_full", {}).get("cagr") if best else None,
                    "best_april_2026_return": best.get("april_2026_return"),
                }
            ),
            flush=True,
        )
        if args.cycles > 0 and cycle >= args.cycles:
            break
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
