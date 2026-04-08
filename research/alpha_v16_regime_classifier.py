#!/usr/bin/env python3
"""
ALPHA ENGINE v16 — REGIME CLASSIFIER RESEARCH
=============================================

Research-only challenger to v9:
- train a simple lagged macro classifier for "next 63d NIFTY return < -5%"
- use that probability to cut risk before price-only crash rules fully trigger
- compare honestly against current production v9 on the same stitched OOS setup

This does NOT change production behavior. It is a clean research harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

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


RESULTS_CACHE = CACHE_DIR / "alpha_v16_regime_classifier_results.json"


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    base: StrategyConfig
    target_horizon: int = 63
    target_drawdown_cutoff: float = -0.05
    feature_lag_days: int = 5
    regime_strength: float = 1.0
    regime_floor: float = 0.20
    daily_cut_threshold: float = 0.35
    alpha: float = 0.10


@dataclass(frozen=True)
class RegimeModel:
    columns: tuple[str, ...]
    means: Dict[str, float]
    scales: Dict[str, float]
    params: Dict[str, float]
    train_size: int
    positive_rate: float


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _last_rank(values: pd.Series) -> float:
        ranked = values.rank(pct=True)
        return float(ranked.iloc[-1])

    return series.rolling(window, min_periods=window).apply(_last_rank, raw=False)


def build_regime_features(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    aligned_macro = macro.reindex(prices.index).ffill()
    breadth_ratio = (prices[RISKY] > prices[RISKY].rolling(200).mean()).sum(axis=1) / len(RISKY)
    mom63 = prices[RISKY].pct_change(63, fill_method=None)
    mom_dispersion = mom63.std(axis=1)

    frame = pd.DataFrame(index=prices.index)
    frame["us10y_d20"] = aligned_macro["US10Y"].diff(20)
    frame["usdinr_21"] = aligned_macro["USDINR"].pct_change(21, fill_method=None)
    frame["crude_63"] = aligned_macro["CRUDE"].pct_change(63, fill_method=None)
    frame["vix_pctile_252"] = rolling_percentile(aligned_macro["INDIAVIX"], 252)
    frame["breadth_min10"] = breadth_ratio.rolling(10).min()
    frame["breadth_mean10"] = breadth_ratio.rolling(10).mean()
    frame["mom_dispersion_21_change"] = mom_dispersion.diff(21)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_regime_target(prices: pd.DataFrame, horizon: int, cutoff: float) -> pd.Series:
    fwd = prices["NIFTY"].shift(-horizon) / prices["NIFTY"] - 1.0
    return (fwd < cutoff).astype(float)


def fit_regime_model(
    train_prices: pd.DataFrame,
    train_macro: pd.DataFrame,
    config: RegimeConfig,
) -> Optional[RegimeModel]:
    features = build_regime_features(train_prices, train_macro).shift(config.feature_lag_days)
    target = build_regime_target(train_prices, config.target_horizon, config.target_drawdown_cutoff)

    frame = features.copy()
    frame["target"] = target
    frame = frame.dropna()
    if len(frame) < 200:
        return None

    X = frame.drop(columns=["target"])
    y = frame["target"]
    if y.nunique() < 2:
        return None

    means = X.mean()
    scales = X.std().replace(0.0, 1.0).fillna(1.0)
    Xz = (X - means) / scales
    Xz = sm.add_constant(Xz, has_constant="add")

    fit = None
    try:
        fit = sm.GLM(y, Xz, family=sm.families.Binomial()).fit(maxiter=200)
    except Exception:
        try:
            fit = sm.Logit(y, Xz).fit_regularized(alpha=config.alpha, L1_wt=0.0, disp=False)
        except Exception:
            return None

    params = {str(k): float(v) for k, v in fit.params.items()}
    return RegimeModel(
        columns=tuple(X.columns),
        means={str(k): float(v) for k, v in means.items()},
        scales={str(k): float(v) for k, v in scales.items()},
        params=params,
        train_size=int(len(frame)),
        positive_rate=float(y.mean()),
    )


def predict_regime_probability(
    model: Optional[RegimeModel],
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    config: RegimeConfig,
) -> pd.Series:
    if model is None:
        return pd.Series(0.0, index=prices.index)

    features = build_regime_features(prices, macro).shift(config.feature_lag_days)
    X = features.reindex(columns=list(model.columns))
    means = pd.Series(model.means)
    scales = pd.Series(model.scales).replace(0.0, 1.0)
    Xz = (X - means) / scales
    Xz = sm.add_constant(Xz, has_constant="add")
    params = pd.Series(model.params)
    Xz = Xz.reindex(columns=params.index, fill_value=0.0)
    linear = Xz.dot(params)
    prob = 1.0 / (1.0 + np.exp(-linear))
    prob = pd.Series(prob, index=prices.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prob.clip(lower=0.0, upper=1.0)


def run_regime_strategy(
    prices: pd.DataFrame,
    regime_prob: pd.Series,
    config: RegimeConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    features = build_features(prices)
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
    regime_prob = regime_prob.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = pd.DataFrame(features["sma50"])
    sma200 = pd.DataFrame(features["sma200"])
    mom63 = pd.DataFrame(features["mom63"])
    mom126 = pd.DataFrame(features["mom126"])
    mom252 = pd.DataFrame(features["mom252"])
    vol63 = pd.DataFrame(features["vol63"])
    rsi14 = pd.DataFrame(features["rsi14"])
    asset_dd20 = pd.DataFrame(features["asset_dd20"])
    asset_vol_ratio = pd.DataFrame(features["asset_vol_ratio"])
    breakout252 = pd.DataFrame(features["breakout252"])
    cash126 = pd.Series(features["cash126"])

    for i, dt in enumerate(prices.index):
        if i < 252:
            weights.iloc[i] = current
            continue

        trend = prices.loc[dt, RISKY] > sma200.loc[dt]
        strong = trend & (prices.loc[dt, RISKY] > sma50.loc[dt]) & (mom126.loc[dt] > cash126.loc[dt])
        breadth = int(strong.sum())
        crash = crash_signal(features, dt)
        p_regime = float(regime_prob.loc[dt])

        if bool(schedule.loc[dt]):
            risk_scale = breadth_risk_scale(breadth)
            if crash:
                risk_scale = min(risk_scale, config.base.crash_floor)

            llm_override = float(override.loc[dt])
            risk_scale = min(risk_scale, 1.0 - 0.65 * llm_override)

            # Regime classifier cuts risk ahead of price-confirmed crashes.
            regime_scale = max(config.regime_floor, 1.0 - config.regime_strength * p_regime)
            risk_scale = min(risk_scale, regime_scale)

            score = 0.20 * (mom63.loc[dt] / vol63.loc[dt].replace(0, np.nan)) + 0.35 * mom126.loc[dt] + 0.45 * mom252.loc[dt]
            score = score.replace([np.inf, -np.inf], np.nan).fillna(-999.0).sort_values(ascending=False)

            rebound_setup = trend & (asset_dd20.loc[dt] <= -0.08) & (rsi14.loc[dt] <= 45.0)
            score = score + rebound_setup.astype(float) * 0.04

            if bool(trend.get("GOLD", False)) and float(asset_vol_ratio.loc[dt].get("GOLD", 0.0) or 0.0) > 1.35:
                score.loc["GOLD"] = float(score.get("GOLD", -999.0)) + 0.03
            if bool(trend.get("SILVER", False)) and bool(breakout252.loc[dt].get("SILVER", False)):
                score.loc["SILVER"] = float(score.get("SILVER", -999.0)) + 0.03

            selected = [asset for asset in score.index if bool(trend.get(asset, False))][: config.base.top_n]

            target = pd.Series(0.0, index=ALL)
            core_w = inverse_vol_weights(vol63.loc[dt], RISKY)
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
                target["GOLD"] += min(0.10, 1.0 - target[RISKY].sum())

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

        if p_regime > config.daily_cut_threshold and current[RISKY].sum() > 0:
            max_risky = max(config.regime_floor, 1.0 - config.regime_strength * p_regime)
            risky_sum = float(current[RISKY].sum())
            if risky_sum > max_risky:
                current = current.copy()
                current[RISKY] *= max_risky / risky_sum
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
    macro: pd.DataFrame,
    runner_factory: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
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

    stitched_returns: List[pd.Series] = []
    stitched_weights: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []

    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        combined_prices = prices.iloc[start_i:end_i]
        combined_macro = macro.reindex(combined_prices.index).ffill()
        test_prices = prices.iloc[mid_i:end_i]
        combined_weights = runner_factory(combined_prices, combined_macro)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(test_prices, test_weights, label, rf=rf, tx_cost=tx_cost)
        rows.append(
            {
                "window": window_id,
                "test_start": str(test_prices.index[0]),
                "test_end": str(test_prices.index[-1]),
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
            "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
            "vol": float(vol),
            "sharpe": float(excess.mean() * 252.0 / vol) if vol > 0 else np.nan,
            "mdd": float(mdd),
            "calmar": float((equity.iloc[-1] ** (1.0 / years) - 1.0) / abs(mdd)) if mdd < 0 else np.nan,
            "turnover": float(turnover),
            "avg_cash": float(weights["CASH"].mean()),
        },
    }


def summarize_bad_windows(
    base_windows: List[Dict[str, object]],
    candidate_windows: List[Dict[str, object]],
    watch_windows: tuple[int, ...] = (1, 7, 10, 14),
) -> List[Dict[str, object]]:
    by_window = {int(row["window"]): row for row in candidate_windows}
    out: List[Dict[str, object]] = []
    for base in base_windows:
        window_id = int(base["window"])
        if window_id not in watch_windows or window_id not in by_window:
            continue
        cand = by_window[window_id]
        out.append(
            {
                "window": window_id,
                "test_start": base["test_start"],
                "test_end": base["test_end"],
                "v9_cagr": float(base["cagr"]),
                "v16_cagr": float(cand["cagr"]),
                "v9_sharpe": float(base["sharpe"]),
                "v16_sharpe": float(cand["sharpe"]),
                "v9_mdd": float(base["mdd"]),
                "v16_mdd": float(cand["mdd"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-classifier challenger against current v9.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000.0
    prices = YahooFinanceSource(universe_mode="benchmark").fetch(args.start, end=args.end, refresh=args.refresh_cache)
    macro = fetch_macro_panel(args.start, refresh=args.refresh_cache).reindex(prices.index).ffill()

    cfg = RegimeConfig(name="v16_regime_classifier", base=BASE_V9)

    def base_runner(frame_prices: pd.DataFrame, _: pd.DataFrame) -> pd.DataFrame:
        return run_strategy(frame_prices, BASE_V9, overlay=None)

    def regime_runner(frame_prices: pd.DataFrame, frame_macro: pd.DataFrame) -> pd.DataFrame:
        train_prices = frame_prices.iloc[:756]
        train_macro = frame_macro.iloc[:756]
        model = fit_regime_model(train_prices, train_macro, cfg)
        regime_prob = predict_regime_probability(model, frame_prices, frame_macro, cfg)
        return run_regime_strategy(frame_prices, regime_prob, cfg, overlay=None)

    full_base = performance_metrics(prices, base_runner(prices, macro), BASE_V9.name, rf=args.rf, tx_cost=tx_cost)
    full_v16 = performance_metrics(prices, regime_runner(prices, macro), cfg.name, rf=args.rf, tx_cost=tx_cost)
    print_results_table("FULL SAMPLE", [full_base, full_v16])

    wfo_base = fixed_config_wfo(prices, macro, base_runner, BASE_V9.name, rf=args.rf, tx_cost=tx_cost)
    wfo_v16 = fixed_config_wfo(prices, macro, regime_runner, cfg.name, rf=args.rf, tx_cost=tx_cost)
    oos_benchmarks = benchmark_oos_metrics(
        prices,
        pd.DatetimeIndex(wfo_base["metrics"]["returns"].index),
        rf=args.rf,
        tx_cost=tx_cost,
    )
    print_results_table("WALK-FORWARD OOS (FIXED CONFIG, STITCHED)", [wfo_base["metrics"], wfo_v16["metrics"], *oos_benchmarks])  # type: ignore[list-item]

    base_returns = pd.Series(wfo_base["metrics"]["returns"])
    v16_returns = pd.Series(wfo_v16["metrics"]["returns"])
    payload = {
        "start": args.start,
        "end": args.end,
        "tx_cost": tx_cost,
        "full_sample": {
            full_base["label"]: {
                **{k: float(v) for k, v in full_base.items() if k not in {"label", "weights", "returns", "equity"}},
                "pretty": format_metrics(full_base),
            },
            full_v16["label"]: {
                **{k: float(v) for k, v in full_v16.items() if k not in {"label", "weights", "returns", "equity"}},
                "pretty": format_metrics(full_v16),
            },
        },
        "wfo": {
            BASE_V9.name: {
                "windows": wfo_base["windows"],
                "metrics": {
                    **{k: float(v) for k, v in wfo_base["metrics"].items() if k not in {"label", "weights", "returns", "equity"}},
                    "pretty": format_metrics(wfo_base["metrics"]),
                },
            },
            cfg.name: {
                "windows": wfo_v16["windows"],
                "metrics": {
                    **{k: float(v) for k, v in wfo_v16["metrics"].items() if k not in {"label", "weights", "returns", "equity"}},
                    "pretty": format_metrics(wfo_v16["metrics"]),
                },
            },
        },
        "pairwise": {
            "v16_minus_v9": hac_difference_test(v16_returns, base_returns),
            "v16_mean_tests": {
                **iid_sharpe_test(v16_returns, args.rf),
                **hac_mean_excess_test(v16_returns, args.rf),
            },
        },
        "bad_window_comparison": summarize_bad_windows(
            wfo_base["windows"],  # type: ignore[arg-type]
            wfo_v16["windows"],  # type: ignore[arg-type]
        ),
    }
    RESULTS_CACHE.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {RESULTS_CACHE}")


if __name__ == "__main__":
    main()
