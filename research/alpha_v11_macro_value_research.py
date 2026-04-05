#!/usr/bin/env python3
"""
ALPHA ENGINE v11 — MACRO + VALUE OVERLAY RESEARCH
=================================================

Design philosophy:
- Keep the strategic composite as the starting point.
- Add only slow-moving overlays that plausibly survive ETF costs.
- Use macro variables to change risk budget, not to micromanage daily trades.
- Use a long-horizon value proxy and 12M momentum to tilt within the risky book.

This is still an honest research script:
- 1-day lag
- 30 bps costs
- monthly execution
- stitched walk-forward OOS
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from market_data.market_store import load_processed_matrix
from strategy.v9_engine import (
    ALL,
    RISKY,
    CACHE_DIR,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    NarrativeOverlay,
    StrategyConfig,
    YahooFinanceSource,
    benchmark_oos_metrics,
    benchmark_weights,
    format_metrics,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
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

MACRO_TICKERS = {
    "INDIAVIX": "^INDIAVIX",
    "USDINR": "INR=X",
    "CRUDE": "CL=F",
    "US10Y": "^TNX",
}

MACRO_CACHE = CACHE_DIR / "macro_value_panel.csv"
RESULTS_CACHE = CACHE_DIR / "alpha_v11_macro_value_results.json"


@dataclass(frozen=True)
class MacroValueConfig:
    name: str
    trade_band: float
    tilt_strength: float
    trend_floor: float
    max_macro_cash_add: float


def fixed_weight_frame(prices: pd.DataFrame, alloc: Dict[str, float], label: str, rf: float, tx_cost: float) -> Dict[str, object]:
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    for asset, weight in alloc.items():
        weights[asset] = weight
    return performance_metrics(prices, weights, label, rf=rf, tx_cost=tx_cost)


def _download_close(ticker: str, start: str) -> pd.Series:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            history = yf.Ticker(ticker).history(start=start, auto_adjust=True, repair=True)
            if history.empty or "Close" not in history.columns:
                raise RuntimeError(f"empty history for {ticker}")
            close = history["Close"].copy()
            if getattr(close.index, "tz", None) is not None:
                close.index = close.index.tz_localize(None)
            return close.sort_index()
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {ticker}: {last_error}")


def fetch_macro_panel(start: str, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not refresh:
        local = load_processed_matrix("india_macro", start=start)
        if local is not None and len(local) > 0:
            return local
    if MACRO_CACHE.exists() and not refresh:
        return pd.read_csv(MACRO_CACHE, index_col=0, parse_dates=True)

    raw = {name: _download_close(ticker, start) for name, ticker in MACRO_TICKERS.items()}
    panel = pd.DataFrame(index=raw["USDINR"].index)
    for name, series in raw.items():
        panel[name] = series
    panel = panel.ffill(limit=5).dropna()
    panel.to_csv(MACRO_CACHE, index=True)
    return panel


def zscore_row(row: pd.Series) -> pd.Series:
    std = row.std()
    if std is None or np.isnan(std) or std == 0:
        return pd.Series(0.0, index=row.index)
    return (row - row.mean()) / std


def build_panel(prices: pd.DataFrame, macro: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    risky_base = pd.Series(COMPOSITE_FIXED_WEIGHTS).reindex(RISKY).astype(float)
    risky_base = risky_base / risky_base.sum()

    breadth = (prices[RISKY] > prices[RISKY].rolling(200).mean()).sum(axis=1) / len(RISKY)
    value_proxy = -prices[RISKY].pct_change(756, fill_method=None)
    value_rank = value_proxy.apply(zscore_row, axis=1).fillna(0.0)
    momentum = prices[RISKY].pct_change(252, fill_method=None)
    momentum_rank = momentum.apply(zscore_row, axis=1).fillna(0.0)
    trend_strength = (prices[RISKY] / prices[RISKY].rolling(200).mean() - 1.0).clip(lower=-0.5, upper=0.5).fillna(0.0)

    aligned_macro = macro.reindex(prices.index).ffill().dropna()
    aligned_macro = aligned_macro.reindex(prices.index).ffill()
    vix_ratio = aligned_macro["INDIAVIX"] / aligned_macro["INDIAVIX"].rolling(126).median() - 1.0
    fx_mom = aligned_macro["USDINR"].pct_change(63, fill_method=None)
    crude_mom = aligned_macro["CRUDE"].pct_change(63, fill_method=None)
    us10y_change = aligned_macro["US10Y"].diff(63)

    return {
        "returns": returns,
        "breadth": breadth.fillna(0.0),
        "value_rank": value_rank,
        "momentum_rank": momentum_rank,
        "trend_strength": trend_strength,
        "vix_ratio": vix_ratio.fillna(0.0),
        "fx_mom": fx_mom.fillna(0.0),
        "crude_mom": crude_mom.fillna(0.0),
        "us10y_change": us10y_change.fillna(0.0),
        "risky_base": risky_base,
    }


def macro_risk_budget(features: Dict[str, pd.DataFrame | pd.Series], dt: pd.Timestamp, config: MacroValueConfig) -> float:
    breadth = float(features["breadth"].loc[dt])  # type: ignore[index]
    vix_ratio = float(features["vix_ratio"].loc[dt])  # type: ignore[index]
    fx_mom = float(features["fx_mom"].loc[dt])  # type: ignore[index]
    crude_mom = float(features["crude_mom"].loc[dt])  # type: ignore[index]
    us10y_change = float(features["us10y_change"].loc[dt])  # type: ignore[index]

    risk = 0.0
    if breadth < 0.34:
        risk += 0.25
    elif breadth < 0.50:
        risk += 0.10

    if vix_ratio > 0.40:
        risk += 0.25
    elif vix_ratio > 0.15:
        risk += 0.10

    if fx_mom > 0.04:
        risk += 0.15
    if crude_mom > 0.15:
        risk += 0.10
    if us10y_change > 5.0:
        risk += 0.10

    return min(config.max_macro_cash_add, risk)


def run_macro_value_strategy(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: MacroValueConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    features = build_panel(prices, macro)
    schedule = schedule_flags(prices.index, "MONTHLY")

    risky_base = features["risky_base"]  # type: ignore[assignment]
    momentum_rank = features["momentum_rank"]  # type: ignore[assignment]
    value_rank = features["value_rank"]  # type: ignore[assignment]
    trend_strength = features["trend_strength"]  # type: ignore[assignment]

    risk_off = overlay.risk_off.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0) if overlay is not None else pd.Series(0.0, index=prices.index)
    asset_bias = overlay.asset_bias.reindex(prices.index).fillna(0.0) if overlay is not None else pd.DataFrame(0.0, index=prices.index, columns=RISKY)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(COMPOSITE_FIXED_WEIGHTS, index=ALL).astype(float)
    current = current / current.sum()

    for i, dt in enumerate(prices.index):
        if i < 756:
            weights.iloc[i] = current
            continue

        if bool(schedule.loc[dt]):
            score = 0.55 * momentum_rank.loc[dt] + 0.45 * value_rank.loc[dt]
            trend_adj = (trend_strength.loc[dt].clip(lower=-0.20, upper=0.20) / 0.20).fillna(0.0)

            target = pd.Series(COMPOSITE_FIXED_WEIGHTS, index=ALL).astype(float)
            risky_weights = risky_base.copy()

            multipliers = 1.0 + config.tilt_strength * score.clip(lower=-1.5, upper=1.5)
            trend_multiplier = config.trend_floor + (1.0 - config.trend_floor) * ((trend_adj + 1.0) / 2.0)
            risky_weights = risky_weights * multipliers * trend_multiplier

            macro_cash_add = macro_risk_budget(features, dt, config)
            llm_cash_add = 0.35 * float(risk_off.loc[dt])
            total_cash_add = min(config.max_macro_cash_add + 0.10, macro_cash_add + llm_cash_add)

            if macro_cash_add > 0.15:
                risky_weights["GOLD"] *= 1.15
                risky_weights["US"] *= 1.05
                risky_weights["MIDCAP"] *= 0.80
                risky_weights["SMALLCAP"] *= 0.70

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            risky_weights = risky_weights * (1.0 + 0.12 * bias_row)

            risky_weights = risky_weights.clip(lower=0.0)
            if risky_weights.sum() <= 0:
                risky_weights = risky_base.copy()
            risky_weights = risky_weights / risky_weights.sum()

            risky_budget = 1.0 - (COMPOSITE_FIXED_WEIGHTS["CASH"] + total_cash_add)
            risky_budget = float(np.clip(risky_budget, 0.50, 0.90))
            for asset in RISKY:
                target[asset] = risky_budget * risky_weights[asset]
            target["CASH"] = 1.0 - target[RISKY].sum()
            target = target / target.sum()

            if float((target - current).abs().max()) > config.trade_band:
                current = target

        weights.iloc[i] = current

    return weights


def parametric_v9_wfo(
    prices: pd.DataFrame,
    candidates: List[StrategyConfig],
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
        return {
            "windows": [],
            "returns": pd.Series(dtype=float),
            "metrics": {
                "label": "v9 WFO Stitched OOS",
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

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        train_prices = prices.iloc[start_i:mid_i]
        test_prices = prices.iloc[mid_i:end_i]

        best_config = candidates[0]
        best_score = -np.inf
        for config in candidates:
            train_overlay = None
            if overlay is not None:
                train_overlay = NarrativeOverlay(
                    risk_off=overlay.risk_off.loc[train_prices.index],
                    asset_bias=overlay.asset_bias.loc[train_prices.index],
                )
            train_weights = run_strategy(train_prices, config, overlay=train_overlay)
            train_metrics = performance_metrics(train_prices, train_weights, config.name, rf=rf, tx_cost=tx_cost)
            score = float(train_metrics["sharpe"]) + 0.25 * float(train_metrics["calmar"]) - 0.05 * float(train_metrics["turnover"])
            if score > best_score:
                best_score = score
                best_config = config

        combined_prices = prices.iloc[start_i:end_i]
        combined_overlay = None
        if overlay is not None:
            combined_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[combined_prices.index],
                asset_bias=overlay.asset_bias.loc[combined_prices.index],
            )
        combined_weights = run_strategy(combined_prices, best_config, overlay=combined_overlay)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices, test_weights, best_config.name, rf=rf, tx_cost=tx_cost)

        picked_rows.append(
            {
                "window": window_id,
                "picked": best_config.name,
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
        "windows": picked_rows,
        "returns": stitched,
        "metrics": {
            "label": "v9 WFO Stitched OOS",
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


def macro_value_wfo(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    candidates: List[MacroValueConfig],
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
        return {
            "windows": [],
            "returns": pd.Series(dtype=float),
            "metrics": {
                "label": "Macro/Value WFO Stitched OOS",
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
            train_weights = run_macro_value_strategy(train_prices, train_macro, cfg, overlay=train_overlay)
            train_metrics = performance_metrics(train_prices, train_weights, cfg.name, rf=rf, tx_cost=tx_cost)
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
        combined_weights = run_macro_value_strategy(combined_prices, combined_macro, best_cfg, overlay=combined_overlay)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices, test_weights, best_cfg.name, rf=rf, tx_cost=tx_cost)

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
            "label": "Macro/Value WFO Stitched OOS",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro/value overlay research versus v9 and strategic composite.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--llm-override-file")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource().fetch(args.start, end=args.end, refresh=args.refresh_cache)
    macro = fetch_macro_panel(args.start, refresh=args.refresh_cache)
    overlay = load_llm_overlay(args.llm_override_file, prices.index)

    v9_cfg = StrategyConfig(
        name="weekly_core85_tilt15",
        execution_frequency="WEEKLY",
        core_weight=0.85,
        tilt_weight=0.15,
        top_n=2,
        trade_band=0.08,
        trade_step=0.75,
        crash_floor=0.70,
    )

    candidates = [
        MacroValueConfig("macro_value_balanced", trade_band=0.05, tilt_strength=0.20, trend_floor=0.70, max_macro_cash_add=0.20),
        MacroValueConfig("macro_value_tilted", trade_band=0.05, tilt_strength=0.30, trend_floor=0.65, max_macro_cash_add=0.20),
        MacroValueConfig("macro_value_defensive", trade_band=0.04, tilt_strength=0.20, trend_floor=0.75, max_macro_cash_add=0.30),
    ]

    print("=" * 88)
    print("ALPHA ENGINE v11 — MACRO + VALUE OVERLAY RESEARCH")
    print("=" * 88)
    print(f"Start date: {args.start}")
    print(f"Transaction cost: {tx_cost:.2%} per trade")
    print("Macro panel: India VIX, USDINR, crude, US 10Y.")
    print("Value proxy: long-horizon reversal across the asset sleeves.")

    full_rows: List[Dict[str, object]] = []
    v9_weights = run_strategy(prices, v9_cfg, overlay=overlay)
    full_rows.append(performance_metrics(prices, v9_weights, v9_cfg.name, rf=args.rf, tx_cost=tx_cost))

    macro_rows: List[Dict[str, object]] = []
    for cfg in candidates:
        weights = run_macro_value_strategy(prices, macro, cfg, overlay=overlay)
        result = performance_metrics(prices, weights, cfg.name, rf=args.rf, tx_cost=tx_cost)
        macro_rows.append(result)
        full_rows.append(result)

    full_rows.extend(
        [
            fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),
            fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),
            performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost),
            performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost),
        ]
    )

    ordered = sorted(full_rows, key=lambda row: (float(row["sharpe"]), float(row["cagr"])), reverse=True)
    print_results_table("FULL SAMPLE", ordered)

    train_days = 756
    test_days = 126
    v9_wfo = parametric_v9_wfo(prices, [v9_cfg], rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=train_days, test_days=test_days)
    macro_wfo = macro_value_wfo(prices, macro, candidates, rf=args.rf, tx_cost=tx_cost, overlay=overlay, train_days=train_days, test_days=test_days)

    if len(v9_wfo["metrics"]["returns"]) > 0:  # type: ignore[index]
        oos_dates = v9_wfo["metrics"]["returns"].index  # type: ignore[index]
        subset = prices.loc[oos_dates]
        oos_benchmarks = [
            fixed_weight_frame(subset, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),
            fixed_weight_frame(subset, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),
            *benchmark_oos_metrics(prices, oos_dates, rf=args.rf, tx_cost=tx_cost),
        ]
        print_results_table(
            "WALK-FORWARD OOS (3Y TRAIN / 6M TEST, STITCHED)",
            [v9_wfo["metrics"], macro_wfo["metrics"], *oos_benchmarks],  # type: ignore[list-item]
        )

    best_macro = max(macro_rows, key=lambda row: (float(row["sharpe"]), float(row["cagr"])))
    composite = next(row for row in full_rows if row["label"] == "Composite Fixed")
    eqwt_risky = next(row for row in full_rows if row["label"] == "EqWt Risky")

    print("\nTakeaway:")
    print(
        f"  Best macro/value full sample: {best_macro['label']} "
        f"CAGR {best_macro['cagr']:.1%}, Sharpe {best_macro['sharpe']:.2f}, "
        f"MaxDD {best_macro['mdd']:.1%}, turnover {best_macro['turnover']:.0%}."
    )
    print(
        f"  Composite fixed benchmark: CAGR {composite['cagr']:.1%}, "
        f"Sharpe {composite['sharpe']:.2f}, MaxDD {composite['mdd']:.1%}."
    )
    print(
        f"  Equal-weight risky benchmark: CAGR {eqwt_risky['cagr']:.1%}, "
        f"Sharpe {eqwt_risky['sharpe']:.2f}, MaxDD {eqwt_risky['mdd']:.1%}."
    )
    print("  If the macro/value overlay does not beat the composite on OOS Sharpe, it should stay in research mode.")

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
            for row in full_rows
        },
        "wfo": {
            "v9": {
                "cagr": v9_wfo["metrics"]["cagr"],
                "sharpe": v9_wfo["metrics"]["sharpe"],
                "mdd": v9_wfo["metrics"]["mdd"],
            },
            "macro_value": {
                "cagr": macro_wfo["metrics"]["cagr"],
                "sharpe": macro_wfo["metrics"]["sharpe"],
                "mdd": macro_wfo["metrics"]["mdd"],
            },
        },
    }
    RESULTS_CACHE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved research summary to {RESULTS_CACHE}")


if __name__ == "__main__":
    main()
