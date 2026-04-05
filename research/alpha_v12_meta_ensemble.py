#!/usr/bin/env python3
"""
ALPHA ENGINE v12 — META ENSEMBLE + LLM RISK HOOK
================================================

Research goal:
- Keep the universe alpha from a strategic composite.
- Add tactical value by blending multiple low-turnover sleeves.
- Let the LLM act as a sparse event-risk governor, not as a return forecaster.

Sleeves blended here:
1. Composite Fixed: strategic long-only benchmark.
2. v9 Low-Turnover Core + Tilt: slow trend/momentum overlay.
3. Canary Momentum: crash-aware dual-momentum sleeve.
4. Macro/Value: slow macro risk-budget + value/momentum tilt.

Meta logic:
- Monthly sleeve selection and sizing.
- Favor sleeves with better trailing momentum, lower volatility, smaller
  recent drawdowns, and lower correlation to the average sleeve.
- Use India VIX and universe breadth to penalize aggressive sleeves in stress.
- Keep a minimum composite floor so the model never drifts too far from the
  strategic benchmark.

This is still an honest research script:
- 1-day lag
- 30 bps costs
- 3-year train / 6-month test walk-forward
- all assets in INR
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
    format_metrics,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    portfolio_returns,
    run_strategy,
    schedule_flags,
)
from research.alpha_v10_canary_research import (
    COMPOSITE_FIXED_WEIGHTS,
    CanaryConfig,
    run_canary_strategy,
)
from research.alpha_v11_macro_value_research import (
    MacroValueConfig,
    fetch_macro_panel,
    macro_value_wfo,
    parametric_v9_wfo,
    run_macro_value_strategy,
)


RESULTS_CACHE = CACHE_DIR / "alpha_v12_meta_ensemble_results.json"
LLM_PACKET_PATH = CACHE_DIR / "alpha_v12_latest_llm_packet.json"


@dataclass(frozen=True)
class MetaConfig:
    name: str
    sleeve_mom_fast: int
    sleeve_mom_slow: int
    trade_band: float
    max_canary_weight: float
    composite_floor: float
    trade_mix: float


BASE_V9 = StrategyConfig(
    name="weekly_core85_tilt15",
    execution_frequency="WEEKLY",
    core_weight=0.85,
    tilt_weight=0.15,
    top_n=2,
    trade_band=0.08,
    trade_step=0.75,
    crash_floor=0.70,
)

BASE_CANARY = CanaryConfig(
    name="canary_midlong_monthly",
    execution_frequency="MONTHLY",
    top_risk=2,
    top_def=2,
    trade_band=0.10,
    score_mode="midlong",
    canary_mode="fractional",
    use_trend=True,
)

BASE_MACRO = MacroValueConfig(
    name="macro_value_tilted",
    trade_band=0.06,
    tilt_strength=0.20,
    trend_floor=0.70,
    max_macro_cash_add=0.20,
)


def fixed_weight_frame(
    prices: pd.DataFrame,
    alloc: Dict[str, float],
    label: str,
    rf: float,
    tx_cost: float,
) -> Dict[str, object]:
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    for asset, weight in alloc.items():
        weights[asset] = weight
    return performance_metrics(prices, weights, label, rf=rf, tx_cost=tx_cost)


def meta_candidates() -> List[MetaConfig]:
    return [
        MetaConfig(
            name="meta_63126_direct",
            sleeve_mom_fast=63,
            sleeve_mom_slow=126,
            trade_band=0.08,
            max_canary_weight=0.20,
            composite_floor=0.25,
            trade_mix=1.00,
        ),
        MetaConfig(
            name="meta_126252_direct",
            sleeve_mom_fast=126,
            sleeve_mom_slow=252,
            trade_band=0.08,
            max_canary_weight=0.20,
            composite_floor=0.30,
            trade_mix=1.00,
        ),
        MetaConfig(
            name="meta_126252_low_canary",
            sleeve_mom_fast=126,
            sleeve_mom_slow=252,
            trade_band=0.08,
            max_canary_weight=0.10,
            composite_floor=0.25,
            trade_mix=1.00,
        ),
        MetaConfig(
            name="meta_63126_smooth",
            sleeve_mom_fast=63,
            sleeve_mom_slow=126,
            trade_band=0.06,
            max_canary_weight=0.20,
            composite_floor=0.25,
            trade_mix=0.50,
        ),
        MetaConfig(
            name="meta_63126_low_canary",
            sleeve_mom_fast=63,
            sleeve_mom_slow=126,
            trade_band=0.08,
            max_canary_weight=0.10,
            composite_floor=0.30,
            trade_mix=1.00,
        ),
    ]


def build_sleeves(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    overlay: Optional[NarrativeOverlay] = None,
) -> Dict[str, pd.DataFrame]:
    sleeves: Dict[str, pd.DataFrame] = {}
    sleeves["composite"] = pd.DataFrame(
        {asset: COMPOSITE_FIXED_WEIGHTS.get(asset, 0.0) for asset in ALL},
        index=prices.index,
    ).fillna(0.0)
    sleeves["v9"] = run_strategy(prices, BASE_V9, overlay=overlay)
    sleeves["canary"] = run_canary_strategy(prices, BASE_CANARY, overlay=overlay)
    sleeves["macro"] = run_macro_value_strategy(prices, macro.reindex(prices.index).ffill(), BASE_MACRO, overlay=overlay)
    return sleeves


def build_meta_inputs(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    sleeves: Dict[str, pd.DataFrame],
    tx_cost: float,
) -> Dict[str, object]:
    sleeve_returns = pd.DataFrame(
        {name: portfolio_returns(prices, weights, tx_cost=tx_cost) for name, weights in sleeves.items()}
    ).dropna()

    aligned_sleeves = {
        name: weights.reindex(sleeve_returns.index).ffill().fillna(0.0)
        for name, weights in sleeves.items()
    }
    aligned_macro = macro.reindex(sleeve_returns.index).ffill()
    aligned_prices = prices.reindex(sleeve_returns.index)

    breadth = (aligned_prices[RISKY] > aligned_prices[RISKY].rolling(200).mean()).sum(axis=1) / len(RISKY)
    vix_ratio = aligned_macro["INDIAVIX"] / aligned_macro["INDIAVIX"].rolling(126).median()

    return {
        "returns": sleeve_returns,
        "weights": aligned_sleeves,
        "breadth": breadth.fillna(0.0),
        "vix_ratio": vix_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0),
    }


def meta_score(subset: pd.DataFrame, config: MetaConfig) -> pd.Series:
    vol = subset.rolling(63).std().iloc[-1] * np.sqrt(252)
    fast_mom = (1.0 + subset.iloc[-config.sleeve_mom_fast :]).prod() - 1.0
    slow_mom = (1.0 + subset.iloc[-config.sleeve_mom_slow :]).prod() - 1.0
    equity = (1.0 + subset).cumprod()
    dd63 = (equity / equity.rolling(63).max() - 1.0).iloc[-1]

    peer_series = subset.mean(axis=1)
    corr = subset.tail(min(126, len(subset))).corrwith(peer_series)
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    score = (
        0.45 * fast_mom
        + 0.55 * slow_mom
        - 0.35 * vol.fillna(vol.median())
        + 0.20 * dd63.fillna(0.0)
        - 0.10 * corr
    )
    return score.replace([np.inf, -np.inf], np.nan).fillna(-1.0)


def current_meta_target(
    dt: pd.Timestamp,
    subset: pd.DataFrame,
    breadth: pd.Series,
    vix_ratio: pd.Series,
    config: MetaConfig,
    llm_risk: float,
) -> pd.Series:
    score = meta_score(subset, config)
    vol = subset.rolling(63).std().iloc[-1] * np.sqrt(252)

    raw = ((score - score.min()) + 0.05) * (1.0 / vol.replace(0.0, np.nan))
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    raw["canary"] = min(raw.get("canary", 0.0), config.max_canary_weight)

    state_vix = float(vix_ratio.loc[dt])
    state_breadth = float(breadth.loc[dt])

    if state_vix > 1.15 or state_breadth < 0.50:
        raw["canary"] *= 0.25
        raw["macro"] *= 0.90
        raw["v9"] *= 1.15
        raw["composite"] *= 1.10

    if state_vix < 0.95 and state_breadth > 0.66:
        raw["canary"] *= 1.15
        raw["macro"] *= 1.05

    if llm_risk > 0:
        raw["canary"] *= max(0.0, 1.0 - 0.90 * llm_risk)
        raw["macro"] *= max(0.0, 1.0 - 0.45 * llm_risk)
        raw["composite"] *= 1.0 + 0.30 * llm_risk
        raw["v9"] *= 1.0 + 0.15 * llm_risk

    raw = raw.clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series({"composite": 0.50, "v9": 0.30, "macro": 0.20, "canary": 0.0})

    target = raw / raw.sum()
    target["composite"] = max(float(target["composite"]), config.composite_floor)
    target["canary"] = min(float(target["canary"]), config.max_canary_weight)
    target = target / target.sum()
    return target


def run_meta_strategy(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: MetaConfig,
    overlay: Optional[NarrativeOverlay] = None,
    tx_cost: float = DEFAULT_TX,
) -> pd.DataFrame:
    sleeves = build_sleeves(prices, macro, overlay=overlay)
    meta_inputs = build_meta_inputs(prices, macro, sleeves, tx_cost=tx_cost)

    sleeve_returns = meta_inputs["returns"]  # type: ignore[assignment]
    sleeve_weights = meta_inputs["weights"]  # type: ignore[assignment]
    breadth = meta_inputs["breadth"]  # type: ignore[assignment]
    vix_ratio = meta_inputs["vix_ratio"]  # type: ignore[assignment]
    schedule = schedule_flags(sleeve_returns.index, "MONTHLY")

    risk_off = (
        overlay.risk_off.reindex(sleeve_returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if overlay is not None
        else pd.Series(0.0, index=sleeve_returns.index)
    )

    current = pd.Series({"composite": 0.50, "v9": 0.30, "macro": 0.20, "canary": 0.0}, dtype=float)
    meta_weights = pd.DataFrame(0.0, index=sleeve_returns.index, columns=current.index)

    min_history = max(config.sleeve_mom_slow, 252)
    for i, dt in enumerate(sleeve_returns.index):
        if i < min_history:
            meta_weights.loc[dt] = current
            continue

        if bool(schedule.loc[dt]):
            subset = sleeve_returns.iloc[: i + 1]
            target = current_meta_target(
                dt=dt,
                subset=subset,
                breadth=breadth,
                vix_ratio=vix_ratio,
                config=config,
                llm_risk=float(risk_off.loc[dt]),
            )
            if float((target - current).abs().max()) > config.trade_band:
                current = config.trade_mix * target + (1.0 - config.trade_mix) * current
                current = current / current.sum()

        meta_weights.loc[dt] = current

    asset_weights = pd.DataFrame(0.0, index=sleeve_returns.index, columns=ALL)
    for sleeve_name, sleeve_frame in sleeve_weights.items():
        asset_weights = asset_weights.add(
            meta_weights[sleeve_name].values.reshape(-1, 1) * sleeve_frame[ALL].values,
            fill_value=0.0,
        )

    asset_weights = asset_weights.div(asset_weights.sum(axis=1), axis=0).fillna(0.0)
    return asset_weights


def meta_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    candidates: List[MetaConfig],
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
                "label": "Meta Ensemble WFO Stitched OOS",
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
            },
        }

    stitched_returns: List[pd.Series] = []
    stitched_weights: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        train_prices = prices.iloc[start_i:mid_i]
        train_macro = macro.reindex(train_prices.index).ffill()
        test_prices = prices.iloc[mid_i:end_i]

        best_cfg = candidates[0]
        best_score = -np.inf
        for cfg in candidates:
            train_overlay = None
            if overlay is not None:
                train_overlay = NarrativeOverlay(
                    risk_off=overlay.risk_off.loc[train_prices.index],
                    asset_bias=overlay.asset_bias.loc[train_prices.index],
                )
            train_weights = run_meta_strategy(train_prices, train_macro, cfg, overlay=train_overlay, tx_cost=tx_cost)
            train_metrics = performance_metrics(train_prices.loc[train_weights.index], train_weights, cfg.name, rf=rf, tx_cost=tx_cost)
            score = float(train_metrics["sharpe"]) + 0.25 * float(train_metrics["calmar"]) - 0.05 * float(train_metrics["turnover"])
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
        combined_weights = run_meta_strategy(combined_prices, combined_macro, best_cfg, overlay=combined_overlay, tx_cost=tx_cost)
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
            "label": "Meta Ensemble WFO Stitched OOS",
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


def build_latest_llm_packet(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    meta_weights: pd.DataFrame,
    asset_weights: pd.DataFrame,
    config: MetaConfig,
    path: Path,
) -> None:
    dt = asset_weights.index[-1]

    risky_prices = prices.reindex(asset_weights.index)[RISKY]
    breadth = float((risky_prices.loc[dt] > risky_prices.rolling(200).mean().loc[dt]).sum() / len(RISKY))
    vix_ratio = float(
        (
            macro.reindex(asset_weights.index).ffill()["INDIAVIX"]
            / macro.reindex(asset_weights.index).ffill()["INDIAVIX"].rolling(126).median()
        ).loc[dt]
    )

    packet = {
        "as_of": dt.strftime("%Y-%m-%d"),
        "model": config.name,
        "instructions": (
            "Use this packet only to decide whether short-lived event-risk overrides are warranted. "
            "Do not forecast returns directly. Prefer sparse, conservative overrides."
        ),
        "macro_state": {
            "breadth_above_200d": breadth,
            "indiavix_ratio_to_6m_median": vix_ratio,
            "usd_inr_3m_return": float(macro.reindex(asset_weights.index).ffill()["USDINR"].pct_change(63).loc[dt]),
            "crude_3m_return": float(macro.reindex(asset_weights.index).ffill()["CRUDE"].pct_change(63).loc[dt]),
            "us10y_3m_change_bps_proxy": float(macro.reindex(asset_weights.index).ffill()["US10Y"].diff(63).loc[dt]),
        },
        "current_meta_weights": {
            sleeve: float(meta_weights.loc[dt, sleeve]) for sleeve in meta_weights.columns
        },
        "current_asset_weights": {
            asset: float(asset_weights.loc[dt, asset]) for asset in ALL
        },
        "output_schema": {
            "default_risk_off_override": "float 0.0..0.6",
            "dates": [
                {
                    "date": "YYYY-MM-DD",
                    "holding_days": "int 1..20",
                    "risk_off_override": "float 0.0..0.6",
                    "asset_bias": {asset: "float -1.0..1.0" for asset in RISKY},
                    "rationale": "short text",
                    "confidence": "float 0.0..1.0",
                }
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2), encoding="utf-8")


def backtest(
    prices: pd.DataFrame,
    params: Optional[Dict[str, object]] = None,
    label: str = "Meta Ensemble",
) -> Dict[str, object]:
    config = meta_candidates()[3]
    if params:
        config = MetaConfig(
            name=str(params.get("name", config.name)),
            sleeve_mom_fast=int(params.get("sleeve_mom_fast", config.sleeve_mom_fast)),
            sleeve_mom_slow=int(params.get("sleeve_mom_slow", config.sleeve_mom_slow)),
            trade_band=float(params.get("trade_band", config.trade_band)),
            max_canary_weight=float(params.get("max_canary_weight", config.max_canary_weight)),
            composite_floor=float(params.get("composite_floor", config.composite_floor)),
            trade_mix=float(params.get("trade_mix", config.trade_mix)),
        )

    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))
    weights = run_meta_strategy(prices, macro, config, overlay=None, tx_cost=DEFAULT_TX)
    return performance_metrics(prices.loc[weights.index], weights, label, rf=DEFAULT_RF, tx_cost=DEFAULT_TX)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-ensemble tactical allocation research.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START, help="Backtest start date.")
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END, help="Backtest end date.")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF, help="Annual risk-free rate.")
    parser.add_argument("--tx-bps", type=float, default=30.0, help="Transaction cost in basis points per trade.")
    parser.add_argument("--refresh-cache", action="store_true", help="Redownload price or macro caches.")
    parser.add_argument("--llm-override-file", help="Optional JSON file with LLM risk-off overrides.")
    parser.add_argument("--export-llm-packet", default=str(LLM_PACKET_PATH), help="Where to write the latest LLM context packet.")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource().fetch(args.start, end=args.end, refresh=args.refresh_cache)
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"), refresh=args.refresh_cache)
    overlay = load_llm_overlay(args.llm_override_file, prices.index)

    candidates = meta_candidates()
    recommended = candidates[3]

    full_sample: List[Dict[str, object]] = []
    meta_outputs: Dict[str, pd.DataFrame] = {}
    for cfg in candidates:
        weights = run_meta_strategy(prices, macro, cfg, overlay=overlay, tx_cost=tx_cost)
        meta_outputs[cfg.name] = weights
        full_sample.append(performance_metrics(prices.loc[weights.index], weights, cfg.name, rf=args.rf, tx_cost=tx_cost))

    v9_result = performance_metrics(
        prices,
        run_strategy(prices, BASE_V9, overlay=overlay),
        BASE_V9.name,
        rf=args.rf,
        tx_cost=tx_cost,
    )
    composite_result = fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost)
    eqwt_all7 = fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost)
    eqwt_risky = performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost)
    nifty = performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost)

    print_results_table(
        "FULL SAMPLE (1-day lag, 30 bps, all INR)",
        [*full_sample, v9_result, composite_result, eqwt_all7, eqwt_risky, nifty],
    )

    meta_wfo = meta_walk_forward(prices, macro, candidates, rf=args.rf, tx_cost=tx_cost, overlay=overlay)
    v9_wfo = parametric_v9_wfo(prices, [BASE_V9], rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=756, test_days=126)
    macro_wfo = macro_value_wfo(prices, macro, [BASE_MACRO], rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=756, test_days=126)

    oos_dates = meta_wfo["metrics"]["returns"].index  # type: ignore[index]
    oos_results: List[Dict[str, object]] = [meta_wfo["metrics"], v9_wfo["metrics"], macro_wfo["metrics"]]  # type: ignore[list-item]
    oos_results.append(fixed_weight_frame(prices.loc[oos_dates], COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost))
    oos_results.append(fixed_weight_frame(prices.loc[oos_dates], {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost))
    oos_results.extend(benchmark_oos_metrics(prices, oos_dates, rf=args.rf, tx_cost=tx_cost))

    print_results_table(
        "WALK-FORWARD OOS (3Y train / 6M test, all INR)",
        oos_results,
    )

    recommended_meta_weights = meta_outputs[recommended.name]
    meta_inputs = build_meta_inputs(prices, macro, build_sleeves(prices, macro, overlay=overlay), tx_cost=tx_cost)
    sleeve_returns = meta_inputs["returns"]  # type: ignore[assignment]
    breadth = meta_inputs["breadth"]  # type: ignore[assignment]
    vix_ratio = meta_inputs["vix_ratio"]  # type: ignore[assignment]
    meta_schedule = schedule_flags(sleeve_returns.index, "MONTHLY")
    meta_history = pd.DataFrame(0.0, index=sleeve_returns.index, columns=["composite", "v9", "macro", "canary"])
    current = pd.Series({"composite": 0.50, "v9": 0.30, "macro": 0.20, "canary": 0.0}, dtype=float)
    llm_risk = (
        overlay.risk_off.reindex(sleeve_returns.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        if overlay is not None
        else pd.Series(0.0, index=sleeve_returns.index)
    )

    for i, dt in enumerate(sleeve_returns.index):
        if i >= max(recommended.sleeve_mom_slow, 252) and bool(meta_schedule.loc[dt]):
            target = current_meta_target(
                dt=dt,
                subset=sleeve_returns.iloc[: i + 1],
                breadth=breadth,
                vix_ratio=vix_ratio,
                config=recommended,
                llm_risk=float(llm_risk.loc[dt]),
            )
            if float((target - current).abs().max()) > recommended.trade_band:
                current = recommended.trade_mix * target + (1.0 - recommended.trade_mix) * current
                current = current / current.sum()
        meta_history.loc[dt] = current

    packet_path = Path(args.export_llm_packet).expanduser().resolve()
    build_latest_llm_packet(
        prices=prices,
        macro=macro,
        meta_weights=meta_history,
        asset_weights=recommended_meta_weights,
        config=recommended,
        path=packet_path,
    )

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
            }
            for row in [*full_sample, v9_result, composite_result, eqwt_all7, eqwt_risky, nifty]
        },
        "wfo": {
            "meta": {
                "cagr": float(meta_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(meta_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(meta_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(meta_wfo["metrics"]["turnover"]),  # type: ignore[index]
            },
            "v9": {
                "cagr": float(v9_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(v9_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(v9_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(v9_wfo["metrics"]["turnover"]),  # type: ignore[index]
            },
            "macro_value": {
                "cagr": float(macro_wfo["metrics"]["cagr"]),  # type: ignore[index]
                "sharpe": float(macro_wfo["metrics"]["sharpe"]),  # type: ignore[index]
                "mdd": float(macro_wfo["metrics"]["mdd"]),  # type: ignore[index]
                "turnover": float(macro_wfo["metrics"]["turnover"]),  # type: ignore[index]
            },
        },
        "packet_path": str(packet_path),
    }
    RESULTS_CACHE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nLatest LLM packet written to:", packet_path)
    print("Results cache written to:", RESULTS_CACHE)


if __name__ == "__main__":
    main()
