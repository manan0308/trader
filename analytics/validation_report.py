#!/usr/bin/env python3
"""
Expanded Validation Pack
========================

This script sits above the strategy code and answers the question:
"How much should we trust these backtest differences?"

It extends the simpler significance script with:
- HAC / Newey-West significance tests
- moving block bootstrap confidence intervals
- Holm multiple-testing adjustment
- coarse calendar block breakdowns
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy import stats

from strategy.v9_engine import ALL, CACHE_DIR, DEFAULT_BACKTEST_END, DEFAULT_BACKTEST_START, DEFAULT_RF, DEFAULT_TX, RISKY, YahooFinanceSource, benchmark_weights, performance_metrics
from research.alpha_v11_macro_value_research import parametric_v9_wfo
from research.alpha_v12_meta_ensemble import BASE_V9
from analytics.significance_report import (
    build_model_returns,
    hac_difference_test,
    hac_mean_excess_test,
    iid_sharpe_test,
    rf_daily,
)


RESULTS_PATH = CACHE_DIR / "validation_pack.json"


def annualized_excess(returns: pd.Series, rf: float) -> float:
    ex = pd.Series(returns).dropna() - rf_daily(rf)
    return float(ex.mean() * 252.0)


def annualized_sharpe(returns: pd.Series, rf: float) -> float:
    series = pd.Series(returns).dropna()
    ex = series - rf_daily(rf)
    vol = float(series.std() * np.sqrt(252.0))
    if vol <= 0:
        return float("nan")
    return float(ex.mean() * 252.0 / vol)


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + pd.Series(returns).dropna()).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")


def cagr(returns: pd.Series) -> float:
    series = pd.Series(returns).dropna()
    if len(series) == 0:
        return float("nan")
    equity = (1.0 + series).cumprod()
    years = len(series) / 252.0
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def bootstrap_sample(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values.copy()
    if block <= 1:
        picks = rng.integers(0, n, size=n)
        return values[picks]

    out: List[float] = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        idx = [(start + offset) % n for offset in range(block)]
        out.extend(values[idx])
    return np.asarray(out[:n], dtype=float)


def bootstrap_ci(
    returns: pd.Series,
    rf: float,
    block: int,
    iterations: int,
    stat_fn: Callable[[pd.Series, float], float],
    seed: int,
) -> Dict[str, float]:
    series = pd.Series(returns).dropna()
    values = series.values.astype(float)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        draw = bootstrap_sample(values, block=block, rng=rng)
        samples.append(stat_fn(pd.Series(draw), rf))
    arr = np.asarray(samples, dtype=float)
    return {
        "point": float(stat_fn(series, rf)),
        "p05": float(np.nanpercentile(arr, 5)),
        "p50": float(np.nanpercentile(arr, 50)),
        "p95": float(np.nanpercentile(arr, 95)),
        "prob_le_zero": float(np.mean(arr <= 0.0)),
    }


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    names = list(p_values)
    vals = [p_values[name] for name in names]
    _, adj, _, _ = multipletests(vals, method="holm")
    return {name: float(value) for name, value in zip(names, adj)}


def sharpe_t_test(returns: pd.Series, rf: float) -> Dict[str, float]:
    series = pd.Series(returns).dropna()
    ex = series - rf_daily(rf)
    daily_sr = float(ex.mean() / ex.std()) if float(ex.std()) > 0 else np.nan
    t_stat = daily_sr * np.sqrt(len(ex)) if np.isfinite(daily_sr) else np.nan
    df = max(len(ex) - 1, 1)
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan
    return {
        "sample_size": float(len(ex)),
        "sharpe": daily_sr * np.sqrt(252.0),
        "sharpe_t_stat": float(t_stat),
        "sharpe_p_value": p_value,
    }


def distribution_diagnostics(returns: pd.Series) -> Dict[str, float | bool]:
    series = pd.Series(returns).dropna()
    skew = float(series.skew()) if len(series) > 2 else float("nan")
    excess_kurt = float(series.kurt()) if len(series) > 3 else float("nan")
    kurtosis = excess_kurt + 3.0 if np.isfinite(excess_kurt) else float("nan")
    return {
        "skewness": skew,
        "kurtosis": kurtosis,
        "excess_kurtosis": excess_kurt,
        "fat_tails": bool(kurtosis > 3.0) if np.isfinite(kurtosis) else False,
    }


def expanding_sharpe_series(returns: pd.Series, rf: float, min_periods: int = 30) -> pd.Series:
    series = pd.Series(returns).dropna()
    ex = series - rf_daily(rf)
    exp_mean = ex.expanding(min_periods=min_periods).mean()
    exp_std = series.expanding(min_periods=min_periods).std()
    out = (exp_mean / exp_std) * np.sqrt(252.0)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def rolling_sharpe_series(returns: pd.Series, rf: float, window: int = 63, min_periods: int = 30) -> pd.Series:
    series = pd.Series(returns).dropna()
    ex = series - rf_daily(rf)
    roll_mean = ex.rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=min_periods).std()
    out = (roll_mean / roll_std) * np.sqrt(252.0)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def sharpe_stability_summary(expanding: pd.Series, rolling: pd.Series) -> Dict[str, float]:
    exp = pd.Series(expanding).dropna()
    rol = pd.Series(rolling).dropna()
    return {
        "expanding_sharpe_latest": float(exp.iloc[-1]) if len(exp) else float("nan"),
        "expanding_sharpe_std": float(exp.std()) if len(exp) > 1 else float("nan"),
        "expanding_sharpe_min": float(exp.min()) if len(exp) else float("nan"),
        "expanding_sharpe_max": float(exp.max()) if len(exp) else float("nan"),
        "rolling_sharpe_latest": float(rol.iloc[-1]) if len(rol) else float("nan"),
        "rolling_sharpe_std": float(rol.std()) if len(rol) > 1 else float("nan"),
        "rolling_sharpe_min": float(rol.min()) if len(rol) else float("nan"),
        "rolling_sharpe_max": float(rol.max()) if len(rol) else float("nan"),
    }


def downsample_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized_freq = "ME" if freq == "M" else freq
    sampled = frame.resample(normalized_freq).last().dropna(how="all")
    if sampled.index[-1] != frame.index[-1]:
        sampled = pd.concat([sampled, frame.iloc[[-1]]]).sort_index()
        sampled = sampled[~sampled.index.duplicated(keep="last")]
    return sampled


def frame_to_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dt, row in frame.iterrows():
        item = {"date": pd.Timestamp(dt).strftime("%Y-%m-%d")}
        for key, value in row.items():
            item[str(key)] = None if pd.isna(value) else float(value)
        rows.append(item)
    return rows


def buy_hold_fixed_mix_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    common_dates: pd.DatetimeIndex,
    base_value: float = 1.0,
) -> pd.Series:
    safe_weights = {asset: float(weight) for asset, weight in weights.items() if asset in prices.columns and weight > 0.0}
    if not safe_weights or len(common_dates) == 0:
        return pd.Series(dtype=float)

    first = prices.loc[common_dates[0], list(safe_weights.keys())]
    units = {asset: (base_value * weight) / float(first[asset]) for asset, weight in safe_weights.items() if float(first[asset]) > 0}
    if not units:
        return pd.Series(dtype=float)

    bh_equity = pd.Series(0.0, index=common_dates, dtype=float)
    for asset, qty in units.items():
        bh_equity = bh_equity.add(prices.loc[common_dates, asset] * qty, fill_value=0.0)
    return bh_equity.pct_change(fill_method=None).fillna(0.0)


def build_asset_portfolio_diagnostics(
    prices: pd.DataFrame,
    rf: float,
    tx_cost: float,
    base_value: float = 1_000_000.0,
) -> Dict[str, object]:
    v9_oos = parametric_v9_wfo(prices, [BASE_V9], rf=rf, tx_cost=tx_cost, overlay=None, train_days=756, test_days=126)
    portfolio_returns = pd.Series(v9_oos["metrics"]["returns"]).dropna()  # type: ignore[index]
    portfolio_weights = pd.DataFrame(v9_oos["metrics"]["weights"]).copy()  # type: ignore[index]
    common_dates = portfolio_returns.index

    price_returns = prices[ALL].pct_change(fill_method=None).reindex(common_dates).fillna(0.0)
    eqwt_metrics = performance_metrics(
        prices.loc[common_dates],
        benchmark_weights(prices.loc[common_dates], "EqWt Risky"),
        "EqWt Risky",
        rf=rf,
        tx_cost=tx_cost,
    )
    eqwt_returns = pd.Series(eqwt_metrics["returns"]).reindex(common_dates).fillna(0.0)  # type: ignore[index]
    buy_hold_weights = {asset: 1.0 / len(RISKY) for asset in RISKY}
    buy_hold_returns = buy_hold_fixed_mix_returns(prices, buy_hold_weights, common_dates)

    series_map: Dict[str, pd.Series] = {asset: price_returns[asset] for asset in ALL}
    series_map["PORTFOLIO_V9"] = portfolio_returns
    series_map["EQWT_RISKY"] = eqwt_returns
    series_map["BUY_HOLD_FIXED_MIX"] = buy_hold_returns

    diagnostics_rows: List[Dict[str, object]] = []
    expanding_frames: Dict[str, pd.Series] = {}
    rolling_frames: Dict[str, pd.Series] = {}
    for name, series in series_map.items():
        expanding = expanding_sharpe_series(series, rf, min_periods=30)
        rolling = rolling_sharpe_series(series, rf, window=63, min_periods=30)
        stability = sharpe_stability_summary(expanding, rolling)
        sharpe_stats = sharpe_t_test(series, rf)
        dist = distribution_diagnostics(series)
        diagnostics_rows.append(
            {
                "name": name,
                "kind": "portfolio" if name == "PORTFOLIO_V9" else ("benchmark" if name in {"EQWT_RISKY", "BUY_HOLD_FIXED_MIX"} else "asset"),
                "annualized_return": cagr(series),
                "annualized_vol": float(pd.Series(series).dropna().std() * np.sqrt(252.0)),
                **sharpe_stats,
                **dist,
                **stability,
            }
        )
        expanding_frames[name] = expanding
        rolling_frames[name] = rolling

    expanding_frame = pd.DataFrame(expanding_frames).dropna(how="all")
    rolling_frame = pd.DataFrame(rolling_frames).dropna(how="all")
    expanding_down = downsample_frame(expanding_frame, "W-FRI")
    rolling_down = downsample_frame(rolling_frame, "W-FRI")

    equity_frame = pd.DataFrame(
        {
            "PORTFOLIO_V9": base_value * (1.0 + portfolio_returns).cumprod(),
            "EQWT_RISKY": base_value * (1.0 + eqwt_returns).cumprod(),
            "BUY_HOLD_FIXED_MIX": base_value * (1.0 + buy_hold_returns).cumprod(),
        }
    )
    equity_down = downsample_frame(equity_frame, "W-FRI")

    alloc_down = downsample_frame(portfolio_weights.reindex(common_dates).ffill().fillna(0.0), "M")

    return {
        "base_value": base_value,
        "benchmark_overlay": "EQWT_RISKY",
        "benchmark_secondary": "BUY_HOLD_FIXED_MIX",
        "buy_hold_fixed_weights": buy_hold_weights,
        "rolling_window_days": 63,
        "expanding_start_days": 30,
        "diagnostics_rows": diagnostics_rows,
        "expanding_sharpe_series": frame_to_rows(expanding_down),
        "rolling_sharpe_series": frame_to_rows(rolling_down),
        "equity_curve_overlay": frame_to_rows(equity_down),
        "allocation_history": frame_to_rows(alloc_down),
        "sample_start": common_dates[0].strftime("%Y-%m-%d") if len(common_dates) else None,
        "sample_end": common_dates[-1].strftime("%Y-%m-%d") if len(common_dates) else None,
    }


def block_table(returns_map: Dict[str, pd.Series], rf: float, freq: str = "3Y") -> List[Dict[str, object]]:
    common_index: pd.DatetimeIndex | None = None
    for series in returns_map.values():
        idx = pd.Series(series).dropna().index
        common_index = idx if common_index is None else common_index.intersection(idx)
    if common_index is None or len(common_index) == 0:
        return []

    frame = pd.DataFrame({name: series.reindex(common_index) for name, series in returns_map.items()}).dropna(how="all")
    if frame.empty:
        return []

    groups = frame.groupby(frame.index.to_period(freq))
    rows: List[Dict[str, object]] = []
    for period, block in groups:
        row: Dict[str, object] = {
            "period": str(period),
            "start": block.index[0].strftime("%Y-%m-%d"),
            "end": block.index[-1].strftime("%Y-%m-%d"),
        }
        for name in frame.columns:
            series = block[name].dropna()
            if len(series) < 20:
                continue
            row[f"{name}_cagr"] = cagr(series)
            row[f"{name}_sharpe"] = annualized_sharpe(series, rf)
            row[f"{name}_mdd"] = max_drawdown(series)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the expanded validation pack on the stitched OOS strategies.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START)
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END)
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--universe-mode", choices=["research", "benchmark", "tradable"], default="benchmark")
    parser.add_argument("--block", type=int, default=21, help="Bootstrap block length in trading days.")
    parser.add_argument("--bootstrap-iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, end=args.end, refresh=args.refresh_cache)
    _, returns_map = build_model_returns(prices, rf=args.rf, tx_cost=args.tx_bps / 10_000)

    model_stats: Dict[str, Dict[str, object]] = {}
    model_pvals: Dict[str, float] = {}
    for name, series in returns_map.items():
        iid = iid_sharpe_test(series, rf=args.rf)
        hac = hac_mean_excess_test(series, rf=args.rf)
        boot_sharpe = bootstrap_ci(
            series,
            rf=args.rf,
            block=args.block,
            iterations=args.bootstrap_iters,
            stat_fn=annualized_sharpe,
            seed=args.seed,
        )
        boot_excess = bootstrap_ci(
            series,
            rf=args.rf,
            block=args.block,
            iterations=args.bootstrap_iters,
            stat_fn=annualized_excess,
            seed=args.seed + 101,
        )
        model_stats[name] = {
            "cagr": cagr(series),
            "mdd": max_drawdown(series),
            **iid,
            **hac,
            "bootstrap_sharpe": boot_sharpe,
            "bootstrap_excess": boot_excess,
        }
        model_pvals[name] = float(hac["hac_p_value"])

    pairwise = {
        "v9_minus_eqwt": hac_difference_test(returns_map["v9"], returns_map["eqwt"]),
        "v9_minus_nifty": hac_difference_test(returns_map["v9"], returns_map["nifty"]),
    }
    pairwise_adj = holm_adjust({name: row["p_value"] for name, row in pairwise.items()})
    model_adj = holm_adjust(model_pvals)
    blocks = block_table({"v9": returns_map["v9"], "eqwt": returns_map["eqwt"], "nifty": returns_map["nifty"]}, rf=args.rf)
    asset_portfolio = build_asset_portfolio_diagnostics(prices, rf=args.rf, tx_cost=args.tx_bps / 10_000, base_value=1_000_000.0)

    print("\nValidation pack summary:")
    for name, row in model_stats.items():
        print(
            f"  {name:<6} "
            f"Sharpe {row['sharpe']:.3f} | "
            f"HAC p {row['hac_p_value']:.4f} (Holm {model_adj[name]:.4f}) | "
            f"Boot Sharpe 5/50/95 "
            f"{row['bootstrap_sharpe']['p05']:.2f}/"
            f"{row['bootstrap_sharpe']['p50']:.2f}/"
            f"{row['bootstrap_sharpe']['p95']:.2f}"
        )

    print("\nPairwise model differences:")
    for name, row in pairwise.items():
        print(
            f"  {name:<14} "
            f"ann diff {row['annualized_return_diff']:.2%} | "
            f"p {row['p_value']:.4f} (Holm {pairwise_adj[name]:.4f})"
        )

    payload = {
        "models": model_stats,
        "pairwise": pairwise,
        "holm_models": model_adj,
        "holm_pairwise": pairwise_adj,
        "calendar_blocks_3y": blocks,
        "asset_portfolio": asset_portfolio,
        "meta": {
            "tx_bps": args.tx_bps,
            "bootstrap_block": args.block,
            "bootstrap_iterations": args.bootstrap_iters,
            "sample_start": prices.index[0].strftime("%Y-%m-%d"),
            "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
            "universe_mode": args.universe_mode,
        },
    }
    Path(RESULTS_PATH).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
