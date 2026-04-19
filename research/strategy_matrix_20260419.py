#!/usr/bin/env python3
"""
Fresh comparison matrix for the main research strategy families.

Outputs:
- cache/strategy_matrix_20260419.csv
- cache/strategy_matrix_20260419.json
- cache/strategy_matrix_20260419_yoy_allocations.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import resolve_cost_model
from research.alpha_v12_meta_ensemble import BASE_V9
from research.alpha_v18_agile_rotation import agile_candidates, blend_weights, cap_and_redistribute, run_agile_rotation_strategy
from research.alpha_v19_momentum_reversion import candidates as v19_candidates
from research.alpha_v19_momentum_reversion import run_momentum_reversion_strategy
from research.alpha_v20_deep_blend_search import BlendSpec as V20BlendSpec
from research.alpha_v20_deep_blend_search import blend_family as v20_blend_family
from research.alpha_v21_price_volume_rotation import (
    apply_rebalance_policy,
    build_mom10_x_vr20_scores,
    build_obv5_scores,
    build_top_n_targets,
)
from research.alpha_v22_iterative_search import CandidateConfig as V22IterativeConfig
from research.alpha_v22_iterative_search import build_features as v22_iterative_features
from research.alpha_v22_iterative_search import build_weights as run_v22_iterative
from research.alpha_v23_jpm_momentum_playbook import paper_variants, run_long_only_paper_variant
from research.alpha_v24_jpm_blend_iterative_search import BlendCandidate as V24BlendCandidate
from research.alpha_v24_jpm_blend_iterative_search import blend_family as v24_blend_family
from research.alpha_v24_jpm_blend_iterative_search import precompute_components as precompute_v24_components
from strategy.v9_engine import ALL, CACHE_DIR, DEFAULT_RF, performance_metrics, run_strategy


BASE_VALUE = 100_000.0
TX_COST = 0.003
COST_MODEL = resolve_cost_model("india_delivery")
APRIL_START = "2026-04-01"
APRIL_END = "2026-04-17"
BROAD_START = "2014-01-01"
ETF_START = "2024-01-01"

ROOT = Path("/Users/mananagarwal/Desktop/2nd brain/plant to image/trader")
HIST_ROOT = Path("/Users/mananagarwal/Desktop/historical data/dhan")
RECENT_ETF = ROOT / "cache" / "dhan_recent_2026_03_18_2026_04_18.csv"

OUTPUT_CSV = ROOT / "cache" / "strategy_matrix_20260419.csv"
OUTPUT_JSON = ROOT / "cache" / "strategy_matrix_20260419.json"
OUTPUT_YOY = ROOT / "cache" / "strategy_matrix_20260419_yoy_allocations.json"

BROAD_FILES = {
    "NIFTY": HIST_ROOT / "NIFTY50_daily.parquet",
    "MIDCAP": HIST_ROOT / "NIFTY_MIDCAP150_daily.parquet",
    "SMALLCAP": HIST_ROOT / "NIFTY_SMLCAP250_daily.parquet",
    "GOLD": HIST_ROOT / "GOLDBEES_daily.parquet",
    "SILVER": HIST_ROOT / "XAGINR_daily.parquet",
    "US": HIST_ROOT / "MON100_daily.parquet",
    "CASH": HIST_ROOT / "LIQUIDBEES_daily.parquet",
}

ETF_FILES = {
    "NIFTY": HIST_ROOT / "etfs" / "NIFTYBEES_daily.parquet",
    "MIDCAP": HIST_ROOT / "etfs" / "MID150BEES_daily.parquet",
    "SMALLCAP": HIST_ROOT / "etfs" / "HDFCSML250_daily.parquet",
    "GOLD": HIST_ROOT / "etfs" / "GOLDBEES_daily.parquet",
    "SILVER": HIST_ROOT / "etfs" / "SILVERBEES_daily.parquet",
    "US": HIST_ROOT / "etfs" / "MON100_daily.parquet",
    "CASH": HIST_ROOT / "etfs" / "LIQUIDBEES_daily.parquet",
}

RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]


def clean_dates(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_localize(None)
    else:
        try:
            if out.dt.tz is not None:
                out = out.dt.tz_localize(None)
        except Exception:
            pass
    return out


def load_panel(file_map: Dict[str, Path], recent_csv: Path | None = None, recent_symbol_map: Dict[str, str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close_map: Dict[str, pd.Series] = {}
    volume_map: Dict[str, pd.Series] = {}
    for asset, path in file_map.items():
        frame = pd.read_parquet(path, columns=["date", "close", "volume"]).copy()
        frame["date"] = clean_dates(frame["date"])
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        close_map[asset] = pd.Series(frame["close"].astype(float).to_numpy(), index=frame["date"], name=asset)
        volume_map[asset] = pd.Series(frame["volume"].fillna(0.0).astype(float).to_numpy(), index=frame["date"], name=asset)

    prices = pd.DataFrame(close_map).sort_index()
    volume = pd.DataFrame(volume_map).sort_index().fillna(0.0)

    if recent_csv is not None and recent_csv.exists() and recent_symbol_map is not None:
        recent = pd.read_csv(recent_csv, parse_dates=["date"])
        recent["date"] = clean_dates(recent["date"])
        recent = recent[recent["symbol"].isin(recent_symbol_map)].copy()
        recent["asset"] = recent["symbol"].map(recent_symbol_map)
        recent_close = recent.pivot(index="date", columns="asset", values="close").sort_index()
        recent_volume = recent.pivot(index="date", columns="asset", values="volume").sort_index().fillna(0.0)
        prices = pd.concat([prices, recent_close], axis=0).sort_index().groupby(level=0).last()
        volume = pd.concat([volume, recent_volume], axis=0).sort_index().groupby(level=0).last().fillna(0.0)

    prices = prices.reindex(columns=ALL).ffill(limit=5).dropna()
    volume = volume.reindex(index=prices.index, columns=ALL).fillna(0.0)
    return prices, volume


def compute_metrics(prices: pd.DataFrame, weights: pd.DataFrame, label: str, start: str, end: str | None = None) -> Dict[str, float]:
    window_prices = prices.loc[pd.Timestamp(start) : pd.Timestamp(end)] if end else prices.loc[pd.Timestamp(start) :]
    window_weights = weights.reindex(window_prices.index).ffill().fillna(0.0)
    metrics = performance_metrics(
        window_prices,
        window_weights,
        label,
        rf=DEFAULT_RF,
        tx_cost=TX_COST,
        cost_model=COST_MODEL,
        base_value=BASE_VALUE,
    )
    return {
        "cagr": float(metrics["cagr"]),
        "vol": float(metrics["vol"]),
        "sharpe": float(metrics["sharpe"]),
        "mdd": float(metrics["mdd"]),
        "calmar": float(metrics["calmar"]),
        "turnover": float(metrics["turnover"]),
        "avg_cash": float(metrics["avg_cash"]),
        "final_value": float(metrics["equity"].iloc[-1] * BASE_VALUE) if len(metrics["equity"]) else float("nan"),
        "return_pct": float(metrics["equity"].iloc[-1] - 1.0) if len(metrics["equity"]) else float("nan"),
    }


def yearly_avg_weights(weights: pd.DataFrame, start: str) -> Dict[str, Dict[str, float]]:
    frame = weights.loc[pd.Timestamp(start) :].copy()
    out: Dict[str, Dict[str, float]] = {}
    for year, group in frame.groupby(frame.index.year):
        out[str(year)] = {asset: float(group[asset].mean()) for asset in ALL}
    return out


def run_v22_simple_reconstructed(prices: pd.DataFrame, *, cash_floor: float = 0.10, cap: float = 0.70, temp: float = 0.05) -> pd.DataFrame:
    m20 = prices[RISKY].pct_change(20)
    m63 = prices[RISKY].pct_change(63)
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0
    risky_budget = 1.0 - cash_floor
    cap_within_risky = cap / risky_budget
    for dt in prices.index:
        if pd.isna(m63.loc[dt]).all():
            weights.loc[dt] = current
            continue
        score = (0.5 * m20.loc[dt] + 0.5 * m63.loc[dt]).replace([np.inf, -np.inf], np.nan).dropna()
        if score.empty:
            target = current.copy()
        else:
            score = score.sort_values(ascending=False)
            shifted = score - float(score.max())
            ex = np.exp(shifted / max(temp, 1e-9))
            probs = ex / float(ex.sum())
            risky_weights = pd.Series(probs, index=score.index, dtype=float)
            risky_weights = cap_and_redistribute(risky_weights, cap_within_risky) * risky_budget
            target = pd.Series(0.0, index=ALL)
            for asset, value in risky_weights.items():
                target[asset] = float(value)
            target["CASH"] = cash_floor
        current = target / float(target.sum())
        weights.loc[dt] = current
    return weights


def build_strategy_weights(
    broad_prices: pd.DataFrame,
    broad_volume: pd.DataFrame,
    etf_prices: pd.DataFrame,
    etf_volume: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    v18_cfg = {cfg.name: cfg for cfg in agile_candidates()}["v18_momentum_heavy_rotation"]
    v19_cfg_bal = {cfg.name: cfg for cfg in v19_candidates()}["v19_balanced_pullback"]
    v19_cfg_cons = {cfg.name: cfg for cfg in v19_candidates()}["v19_conservative_pullback"]
    jpm_cfgs = {cfg.name: cfg for cfg in paper_variants()}

    v22_iter_cfg = V22IterativeConfig(
        frequency="DAILY",
        top_n=2,
        temp=0.1,
        step=0.5,
        band=0.0,
        w5=0.14152937512071448,
        w10=0.18918168682968875,
        w20=0.40303229610853425,
        w63=0.26625664194106247,
        pre_pos20_min=-0.05,
        pre_pos63_min=-0.05,
        post_pos20_min=-0.02,
        post_pos63_min=-0.1,
        invest1=0.05,
        invest2=0.4,
        invest3=0.75,
        thr1=0.02,
        thr2=0.05,
        thr3=0.1,
        max_asset=0.5,
        us_cap=0.7,
        us_bias=0.02,
        gold_bias=0.04,
        silver_bias=0.01,
        gold_shift=0.05,
        gold_m20_min=-0.02,
        scout_size=0.01,
        scout_m5_min=0.01,
        scout_m20_min=0.0,
        nifty_floor=0.02,
    )

    v20_spec = V20BlendSpec(
        name="blend_80v9_19v19_conservative_pullback_b050_s075",
        weights={"v9": 0.8, "v19_conservative_pullback": 0.2},
        band=0.05,
        step=0.75,
    )
    v24_candidate = V24BlendCandidate(
        weights={"v9": 0.4, "v23_jpm_scorecard8_top3": 0.402, "v23_jpm_dtf3signal_top3": 0.198},
        band=0.025,
        step=1.0,
    )

    def v9(frame: pd.DataFrame) -> pd.DataFrame:
        return run_strategy(frame, BASE_V9, overlay=None).reindex(columns=ALL).fillna(0.0)

    def v18_7030(frame: pd.DataFrame) -> pd.DataFrame:
        return blend_weights(v9(frame), run_agile_rotation_strategy(frame, v18_cfg, overlay=None), 0.70)

    def v19_7030(frame: pd.DataFrame) -> pd.DataFrame:
        return blend_weights(v9(frame), run_momentum_reversion_strategy(frame, v19_cfg_bal, overlay=None), 0.70)

    def v20_shiny(frame: pd.DataFrame) -> pd.DataFrame:
        base = {
            "v9": v9(frame),
            "v19_conservative_pullback": run_momentum_reversion_strategy(frame, v19_cfg_cons, overlay=None).reindex(columns=ALL).fillna(0.0),
        }
        return v20_blend_family(base, v20_spec)

    def v21_obv_daily(frame: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        targets = build_top_n_targets(build_obv5_scores(frame, volume), top_n=2)
        return apply_rebalance_policy(targets, frequency="DAILY", trade_band=0.0, trade_step=1.0)

    def v21_obv_weekly(frame: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        targets = build_top_n_targets(build_obv5_scores(frame, volume), top_n=2)
        return apply_rebalance_policy(targets, frequency="WEEKLY", trade_band=0.20, trade_step=0.50)

    def v21_momvol_daily(frame: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        targets = build_top_n_targets(build_mom10_x_vr20_scores(frame, volume), top_n=2)
        return apply_rebalance_policy(targets, frequency="DAILY", trade_band=0.0, trade_step=1.0)

    def v21_momvol_weekly(frame: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        targets = build_top_n_targets(build_mom10_x_vr20_scores(frame, volume), top_n=2)
        return apply_rebalance_policy(targets, frequency="WEEKLY", trade_band=0.20, trade_step=0.50)

    def v22_iter(frame: pd.DataFrame) -> pd.DataFrame:
        return run_v22_iterative(frame, v22_iterative_features(frame), v22_iter_cfg)

    def v24(frame: pd.DataFrame) -> pd.DataFrame:
        components = precompute_v24_components(frame)
        return v24_blend_family(components, v24_candidate)

    broad_v9 = v9(broad_prices)
    etf_v9 = v9(etf_prices)
    broad_v18 = v18_7030(broad_prices)
    etf_v18 = v18_7030(etf_prices)
    broad_v19 = v19_7030(broad_prices)
    etf_v19 = v19_7030(etf_prices)
    broad_v20 = v20_shiny(broad_prices)
    etf_v20 = v20_shiny(etf_prices)
    broad_v21_obv_d = v21_obv_daily(broad_prices, broad_volume)
    etf_v21_obv_d = v21_obv_daily(etf_prices, etf_volume)
    broad_v21_obv_w = v21_obv_weekly(broad_prices, broad_volume)
    etf_v21_obv_w = v21_obv_weekly(etf_prices, etf_volume)
    broad_v21_mv_d = v21_momvol_daily(broad_prices, broad_volume)
    etf_v21_mv_d = v21_momvol_daily(etf_prices, etf_volume)
    broad_v21_mv_w = v21_momvol_weekly(broad_prices, broad_volume)
    etf_v21_mv_w = v21_momvol_weekly(etf_prices, etf_volume)
    broad_v22_simple = run_v22_simple_reconstructed(broad_prices)
    etf_v22_simple = run_v22_simple_reconstructed(etf_prices)
    broad_v22_iter = v22_iter(broad_prices)
    etf_v22_iter = v22_iter(etf_prices)
    broad_v24 = v24(broad_prices)
    etf_v24 = v24(etf_prices)

    return {
        "v18 70% v9 + 30% momentum": {
            "english_name": "70% v9 core plus 30% aggressive relative-strength rotation",
            "broad_weights": broad_v18,
            "etf_weights": etf_v18,
            "notes": "",
        },
        "v19 70% v9 + 30% pullback": {
            "english_name": "70% v9 core plus 30% trend-safe pullback buyer",
            "broad_weights": broad_v19,
            "etf_weights": etf_v19,
            "notes": "",
        },
        "Shiny ETF v20 fixed spec": {
            "english_name": "80% v9 plus 20% conservative pullback blend with trade band smoothing",
            "broad_weights": broad_v20,
            "etf_weights": etf_v20,
            "notes": "",
        },
        "v21_mom10_x_vr20_top2": {
            "english_name": "Daily top-2 rotation on 10-day momentum times 20-day relative volume",
            "broad_weights": broad_v21_mv_d,
            "etf_weights": etf_v21_mv_d,
            "notes": "",
        },
        "v9_quant": {
            "english_name": "Baseline v9 diversified tactical multi-asset allocator",
            "broad_weights": broad_v9,
            "etf_weights": etf_v9,
            "notes": "",
        },
        "v21_obv5_top2": {
            "english_name": "Daily top-2 rotation on 5-day OBV acceleration",
            "broad_weights": broad_v21_obv_d,
            "etf_weights": etf_v21_obv_d,
            "notes": "",
        },
        "v21_mom10_x_vr20_top2_weekly_s050_b020": {
            "english_name": "Weekly slowed momentum-times-volume top-2 rotation",
            "broad_weights": broad_v21_mv_w,
            "etf_weights": etf_v21_mv_w,
            "notes": "",
        },
        "v21_obv5_top2_weekly_s050_b020": {
            "english_name": "Weekly slowed OBV top-2 rotation",
            "broad_weights": broad_v21_obv_w,
            "etf_weights": etf_v21_obv_w,
            "notes": "",
        },
        "v22_simple_10cash_70cap": {
            "english_name": "Reconstructed simple capped momentum model with 10% cash floor and 70% single-asset cap",
            "broad_weights": broad_v22_simple,
            "etf_weights": etf_v22_simple,
            "notes": "reconstructed_from_research_notes",
        },
        "v22_daily_top2_u70_g5_mx50_sc10": {
            "english_name": "Friend-style daily top-2 momentum model with US bias, gold shift, and small India scout",
            "broad_weights": broad_v22_iter,
            "etf_weights": etf_v22_iter,
            "notes": "",
        },
        "v24_20v23_jpm_dtf3signal_top3_40v23_jpm_scorecard8_top3_40v9_b025_s100": {
            "english_name": "Current best JPM-inspired blend of v9, scorecard momentum, and diversified trend following",
            "broad_weights": broad_v24,
            "etf_weights": etf_v24,
            "notes": "",
        },
    }


def main() -> None:
    broad_prices, broad_volume = load_panel(BROAD_FILES)
    etf_prices, etf_volume = load_panel(
        ETF_FILES,
        recent_csv=RECENT_ETF,
        recent_symbol_map={
            "NIFTYBEES": "NIFTY",
            "MID150BEES": "MIDCAP",
            "HDFCSML250": "SMALLCAP",
            "GOLDBEES": "GOLD",
            "SILVERBEES": "SILVER",
            "MON100": "US",
            "LIQUIDBEES": "CASH",
        },
    )

    strategies = build_strategy_weights(broad_prices, broad_volume, etf_prices, etf_volume)

    rows = []
    detail_payload: Dict[str, object] = {}
    yoy_payload: Dict[str, object] = {}

    for strategy_name, payload in strategies.items():
        broad_weights = payload["broad_weights"]
        etf_weights = payload["etf_weights"]
        broad_metrics = compute_metrics(broad_prices, broad_weights, strategy_name, BROAD_START, "2026-04-10")
        etf_metrics = compute_metrics(etf_prices, etf_weights, strategy_name, ETF_START, APRIL_END)
        april_metrics = compute_metrics(etf_prices, etf_weights, strategy_name, APRIL_START, APRIL_END)

        rows.append(
            {
                "strategy_name": strategy_name,
                "english_name": payload["english_name"],
                "broad_cagr_pct": round(100.0 * broad_metrics["cagr"], 2),
                "broad_sharpe": round(broad_metrics["sharpe"], 3),
                "broad_maxdd_pct": round(100.0 * broad_metrics["mdd"], 2),
                "broad_final_1lakh": round(broad_metrics["final_value"], 2),
                "broad_turnover_pct": round(100.0 * broad_metrics["turnover"], 1),
                "broad_avg_cash_pct": round(100.0 * broad_metrics["avg_cash"], 1),
                "etf_cagr_pct": round(100.0 * etf_metrics["cagr"], 2),
                "etf_sharpe": round(etf_metrics["sharpe"], 3),
                "etf_maxdd_pct": round(100.0 * etf_metrics["mdd"], 2),
                "etf_final_1lakh": round(etf_metrics["final_value"], 2),
                "etf_turnover_pct": round(100.0 * etf_metrics["turnover"], 1),
                "etf_avg_cash_pct": round(100.0 * etf_metrics["avg_cash"], 1),
                "april_2026_return_pct": round(100.0 * april_metrics["return_pct"], 2),
                "notes": payload["notes"],
            }
        )

        detail_payload[strategy_name] = {
            "english_name": payload["english_name"],
            "notes": payload["notes"],
            "broad_metrics": broad_metrics,
            "etf_metrics": etf_metrics,
            "april_2026_metrics": april_metrics,
            "broad_last_weights": {asset: float(broad_weights[asset].iloc[-1]) for asset in ALL},
            "etf_last_weights": {asset: float(etf_weights[asset].iloc[-1]) for asset in ALL},
        }
        yoy_payload[strategy_name] = {
            "broad_avg_weights_by_year": yearly_avg_weights(broad_weights, BROAD_START),
            "etf_avg_weights_by_year": yearly_avg_weights(etf_weights, ETF_START),
        }

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")
    OUTPUT_YOY.write_text(json.dumps(yoy_payload, indent=2), encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved YOY: {OUTPUT_YOY}")


if __name__ == "__main__":
    main()
