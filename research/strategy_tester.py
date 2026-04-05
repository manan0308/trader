#!/usr/bin/env python3
"""
Strategy Tester
===============

Minimal-adaptation harness for evaluating pasted allocation scripts against the
shared INR universe and common benchmarks.

Supported patterns:
- module exposes `backtest(prices, params=None, label="...")`
- module exposes `bt(prices, ...)`
- module exposes `run_strategy(prices, ...)` or `generate_signals(prices, ...)`
  returning a weights DataFrame
- module exposes `backtest(prices, allocator_class, params, label)` and contains
  a class such as `Allocator` or `HonestAllocator`

This tester standardizes metrics where possible:
- 1-day lag
- 30 bps default transaction cost
- CAGR / Vol / Sharpe / MaxDD / Calmar / turnover
- fixed benchmark set including a 7-sleeve composite
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.v9_engine import (
    ALL,
    RISKY,
    DEFAULT_BACKTEST_END,
    DEFAULT_BACKTEST_START,
    DEFAULT_RF,
    benchmark_weights,
    performance_metrics,
    YahooFinanceSource,
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

BACKTEST_FN_NAMES = ["backtest", "bt", "backtest_v10", "backtest_v11"]
WEIGHT_FN_NAMES = ["run_strategy", "generate_signals", "generate_weights", "weights"]
ALLOCATOR_CLASS_NAMES = ["Allocator", "HonestAllocator", "OmegaAllocator"]


def load_module(path: str):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, module_path


def list_callables(module) -> Dict[str, str]:
    result = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if callable(obj):
            try:
                result[name] = str(inspect.signature(obj))
            except Exception:
                result[name] = "(signature unavailable)"
    return result


def fixed_weight_frame(prices: pd.DataFrame, alloc: Dict[str, float], label: str, rf: float, tx_cost: float) -> Dict[str, object]:
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    for asset, weight in alloc.items():
        weights[asset] = weight
    return performance_metrics(prices, weights, label, rf=rf, tx_cost=tx_cost)


def metrics_from_returns(returns: pd.Series, label: str, rf: float) -> Dict[str, object]:
    series = returns.dropna().astype(float)
    equity = (1.0 + series).cumprod()
    years = len(series) / 252
    rf_daily = (1.0 + rf) ** (1 / 252) - 1.0
    excess = series - rf_daily
    vol = series.std() * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1.0).min()
    cagr = equity.iloc[-1] ** (1 / years) - 1.0
    return {
        "label": label,
        "weights": None,
        "returns": series,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": excess.mean() * 252 / vol if vol > 0 else np.nan,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "turnover": np.nan,
        "avg_cash": np.nan,
    }


def format_metrics(row: Dict[str, object]) -> Dict[str, str]:
    def pct(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.1%}"

    def num(v):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}"

    return {
        "CAGR": pct(row.get("cagr")),
        "Vol": pct(row.get("vol")),
        "Sharpe": num(row.get("sharpe")),
        "MaxDD": pct(row.get("mdd")),
        "Calmar": num(row.get("calmar")),
        "AnnTurn": pct(row.get("turnover")),
        "AvgCash": pct(row.get("avg_cash")),
    }


def _extract_allocator_class(module, fn: Callable) -> Optional[type]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return None
    if "allocator_class" not in sig.parameters:
        return None
    for name in ALLOCATOR_CLASS_NAMES:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def call_function(fn: Callable, module, prices: pd.DataFrame, params: Dict[str, Any], label: str, rf: float, tx_cost: float):
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        if name in ("prices", "data", "df"):
            kwargs[name] = prices
        elif name == "config" and hasattr(module, "StrategyConfig") and params:
            kwargs[name] = getattr(module, "StrategyConfig")(**params)
        elif name in ("params", "p", "config"):
            kwargs[name] = params
        elif name in ("label", "name"):
            kwargs[name] = label
        elif name in ("rf", "rf_annual"):
            kwargs[name] = rf
        elif name in ("tx_cost", "tx", "cost"):
            kwargs[name] = tx_cost
        elif name == "allocator_class":
            allocator_class = _extract_allocator_class(module, fn)
            if allocator_class is not None:
                kwargs[name] = allocator_class
    return fn(**kwargs)


def normalize_result(result: Any, prices: pd.DataFrame, label: str, rf: float, tx_cost: float) -> Dict[str, object]:
    if isinstance(result, pd.DataFrame):
        return performance_metrics(prices, result[ALL], label, rf=rf, tx_cost=tx_cost)

    if isinstance(result, dict):
        if "weights" in result and isinstance(result["weights"], pd.DataFrame):
            weights = result["weights"].reindex(prices.index).ffill().fillna(0.0)
            return performance_metrics(prices, weights[ALL], label, rf=rf, tx_cost=tx_cost)

        if "returns" in result and isinstance(result["returns"], pd.Series):
            return metrics_from_returns(result["returns"], label, rf=rf)

        metrics = result.get("metrics") or result.get("m")
        if isinstance(metrics, dict):
            standardized = {
                "label": label,
                "weights": result.get("weights"),
                "returns": result.get("returns"),
                "equity": result.get("equity"),
                "cagr": metrics.get("CAGR", metrics.get("cagr", result.get("_cagr"))),
                "vol": metrics.get("Vol", metrics.get("vol")),
                "sharpe": metrics.get("Sharpe", metrics.get("sharpe", result.get("_sh"))),
                "mdd": metrics.get("MaxDD", metrics.get("mdd", result.get("_mdd"))),
                "calmar": metrics.get("Calmar", metrics.get("calmar", result.get("_cal"))),
                "turnover": metrics.get("Turnover", metrics.get("turnover", result.get("_turn"))),
                "avg_cash": result.get("_ct", metrics.get("Cash%", metrics.get("avg_cash"))),
            }
            for key in ("cagr", "vol", "sharpe", "mdd", "calmar", "turnover", "avg_cash"):
                val = standardized[key]
                if isinstance(val, str):
                    standardized[key] = np.nan
            return standardized

    raise RuntimeError("Unsupported strategy output. Expose a weights DataFrame, a returns Series, or a standard backtest dict.")


def autodetect_callable(module, explicit_name: Optional[str]) -> Tuple[Callable, str]:
    if explicit_name:
        if not hasattr(module, explicit_name):
            raise RuntimeError(f"Callable `{explicit_name}` not found in module.")
        return getattr(module, explicit_name), explicit_name

    for name in BACKTEST_FN_NAMES + WEIGHT_FN_NAMES:
        if hasattr(module, name) and callable(getattr(module, name)):
            return getattr(module, name), name
    raise RuntimeError("No supported callable found. Use `--list-callables` or `--callable`.")


def run_strategy_file(
    strategy_path: str,
    prices: pd.DataFrame,
    params: Dict[str, Any],
    label: str,
    rf: float,
    tx_cost: float,
    callable_name: Optional[str] = None,
) -> Tuple[Dict[str, object], str]:
    module, _ = load_module(strategy_path)
    fn, fn_name = autodetect_callable(module, callable_name)
    raw = call_function(fn, module, prices, params, label, rf, tx_cost)
    normalized = normalize_result(raw, prices, label, rf, tx_cost)
    return normalized, fn_name


def stitched_wfo(
    strategy_path: str,
    params: Dict[str, Any],
    rf: float,
    tx_cost: float,
    callable_name: Optional[str],
    start: str,
) -> Optional[Dict[str, object]]:
    prices = YahooFinanceSource().fetch(start)
    train_days = 504
    test_days = 126
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    if not windows:
        return None

    stitched: list[pd.Series] = []
    for start_i, mid_i, end_i in windows:
        combined = prices.iloc[start_i:end_i]
        test_prices = prices.iloc[mid_i:end_i]
        result, _ = run_strategy_file(strategy_path, combined, params=params, label="WFO", rf=rf, tx_cost=tx_cost, callable_name=callable_name)
        returns = result.get("returns")
        weights = result.get("weights")
        if isinstance(weights, pd.DataFrame):
            test_weights = weights.reindex(test_prices.index).ffill().fillna(0.0)
            test_result = performance_metrics(test_prices, test_weights[ALL], "WFO Window", rf=rf, tx_cost=tx_cost)
            stitched.append(test_result["returns"])  # type: ignore[arg-type]
        elif isinstance(returns, pd.Series):
            stitched.append(returns.reindex(test_prices.index).dropna())

    if not stitched:
        return None

    stitched_returns = pd.concat(stitched).sort_index()
    stitched_returns = stitched_returns[~stitched_returns.index.duplicated(keep="last")]
    return metrics_from_returns(stitched_returns, "Stitched WFO", rf=rf)


def print_table(rows: Iterable[Dict[str, object]], title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    headers = ["Label", "CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "AnnTurn", "AvgCash"]
    widths = {"Label": 28, "CAGR": 9, "Vol": 9, "Sharpe": 8, "MaxDD": 9, "Calmar": 8, "AnnTurn": 9, "AvgCash": 9}
    print(" ".join(f"{h:>{widths[h]}}" for h in headers))
    print("-" * 100)
    for row in rows:
        fmt = format_metrics(row)
        print(
            " ".join(
                [
                    f"{str(row['label']):>{widths['Label']}}",
                    f"{fmt['CAGR']:>{widths['CAGR']}}",
                    f"{fmt['Vol']:>{widths['Vol']}}",
                    f"{fmt['Sharpe']:>{widths['Sharpe']}}",
                    f"{fmt['MaxDD']:>{widths['MaxDD']}}",
                    f"{fmt['Calmar']:>{widths['Calmar']}}",
                    f"{fmt['AnnTurn']:>{widths['AnnTurn']}}",
                    f"{fmt['AvgCash']:>{widths['AvgCash']}}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pasted strategy script on the shared INR universe.")
    parser.add_argument("strategy_path", nargs="?", help="Path to the strategy .py file to evaluate.")
    parser.add_argument("--strategy", dest="strategy_path_flag", help="Path to the strategy .py file to evaluate.")
    parser.add_argument("--callable", "--strategy-fn", dest="callable_name", help="Optional callable name if auto-detect fails.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START, help="Backtest start date for the shared price set.")
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END, help="Backtest end date for the shared price set.")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--label", default="Candidate Strategy")
    parser.add_argument("--params-json", "--params", dest="params_json", help="Inline JSON object of strategy params.")
    parser.add_argument("--list-callables", action="store_true", help="List importable callables and exit.")
    parser.add_argument("--skip-wfo", action="store_true", help="Skip stitched walk-forward evaluation.")
    parser.add_argument("--force-standard-data", action="store_true", help="Accepted for convenience. Shared cleaned INR data is already the default.")
    parser.add_argument("--save-json", "--output-json", dest="save_json", help="Optional path to save the summarized results as JSON.")
    args = parser.parse_args()

    strategy_path = args.strategy_path_flag or args.strategy_path
    if not strategy_path:
        parser.error("Provide a strategy file via positional `strategy_path` or `--strategy`.")

    module, module_path = load_module(strategy_path)
    if args.list_callables:
        print(f"Callables in {module_path}:")
        for name, sig in list_callables(module).items():
            print(f"  {name}{sig}")
        return

    params = json.loads(args.params_json) if args.params_json else {}
    tx_cost = args.tx_bps / 10_000
    prices = YahooFinanceSource().fetch(args.start, end=args.end)

    candidate, fn_name = run_strategy_file(
        strategy_path,
        prices,
        params=params,
        label=args.label,
        rf=args.rf,
        tx_cost=tx_cost,
        callable_name=args.callable_name,
    )

    rows = [
        candidate,
        fixed_weight_frame(prices, COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),
        fixed_weight_frame(prices, {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),
        performance_metrics(prices, benchmark_weights(prices, "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost),
        performance_metrics(prices, benchmark_weights(prices, "Nifty B&H"), "Nifty B&H", rf=args.rf, tx_cost=tx_cost),
        performance_metrics(prices, benchmark_weights(prices, "60/40 Nifty/Cash"), "60/40 Nifty/Cash", rf=args.rf, tx_cost=tx_cost),
    ]
    rows = sorted(rows, key=lambda row: (float(row.get("sharpe", np.nan)), float(row.get("cagr", np.nan))), reverse=True)
    print(f"Imported callable: {fn_name}")
    print_table(rows, "FULL SAMPLE")

    wfo_row = None
    if not args.skip_wfo:
        wfo_row = stitched_wfo(strategy_path, params=params, rf=args.rf, tx_cost=tx_cost, callable_name=args.callable_name, start=args.start)
        if wfo_row is not None:
            wfo_rows = [
                wfo_row,
                fixed_weight_frame(prices.loc[wfo_row["returns"].index], COMPOSITE_FIXED_WEIGHTS, "Composite Fixed", rf=args.rf, tx_cost=tx_cost),  # type: ignore[index]
                fixed_weight_frame(prices.loc[wfo_row["returns"].index], {asset: 1.0 / len(ALL) for asset in ALL}, "EqWt All 7", rf=args.rf, tx_cost=tx_cost),  # type: ignore[index]
                performance_metrics(prices.loc[wfo_row["returns"].index], benchmark_weights(prices.loc[wfo_row["returns"].index], "EqWt Risky"), "EqWt Risky", rf=args.rf, tx_cost=tx_cost),  # type: ignore[index]
            ]
            print_table(wfo_rows, "STITCHED WFO")

    if args.save_json:
        payload = {
            "strategy": str(module_path),
            "callable": fn_name,
            "params": params,
            "full_sample": {
                row["label"]: {
                    "cagr": row.get("cagr"),
                    "vol": row.get("vol"),
                    "sharpe": row.get("sharpe"),
                    "mdd": row.get("mdd"),
                    "calmar": row.get("calmar"),
                    "turnover": row.get("turnover"),
                    "avg_cash": row.get("avg_cash"),
                }
                for row in rows
            },
            "wfo": None if wfo_row is None else {
                "cagr": wfo_row.get("cagr"),
                "vol": wfo_row.get("vol"),
                "sharpe": wfo_row.get("sharpe"),
                "mdd": wfo_row.get("mdd"),
                "calmar": wfo_row.get("calmar"),
            },
        }
        out = Path(args.save_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved summary to {out}")


if __name__ == "__main__":
    main()
