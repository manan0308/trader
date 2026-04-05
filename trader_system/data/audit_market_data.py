#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from trader_system.data.market_store import (
    DATA_ROOT,
    INDIA_BENCHMARK_TICKERS,
    INDIA_MACRO_TICKERS,
    INDIA_RESEARCH_TICKERS,
    INDIA_TRADABLE_TICKERS,
    RAW_DIR,
    US_ANALOG_UNIVERSES,
    dataset_names,
    load_processed_matrix,
    load_raw_bars,
)
from trader_system.runtime.store import write_json


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_PATH = CACHE_DIR / "data_market_audit.json"

DATASET_TICKERS: Dict[str, Dict[str, str]] = {
    "india_research": INDIA_RESEARCH_TICKERS,
    "india_benchmark": INDIA_BENCHMARK_TICKERS,
    "india_tradable": INDIA_TRADABLE_TICKERS,
    "india_macro": INDIA_MACRO_TICKERS,
    "us_analog_shy": US_ANALOG_UNIVERSES["us_analog_shy"],
    "us_analog_ief": US_ANALOG_UNIVERSES["us_analog_ief"],
    "us_analog_bil": US_ANALOG_UNIVERSES["us_analog_bil"],
}


def max_stale_run(series: pd.Series) -> int:
    flat = series.diff().fillna(1).eq(0)
    if not flat.any():
        return 0
    groups = flat.ne(flat.shift()).cumsum()
    return int(flat.groupby(groups).sum().max())


def spike_threshold(asset: str) -> float:
    if asset == "CASH":
        return 0.02
    if asset in {"GOLD", "SILVER", "CRUDE", "INDIAVIX", "US10Y"}:
        return 0.20
    return 0.15


def degenerate_repaired_jumps(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "Repaired?" not in frame.columns or not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        return []
    repaired = frame["Repaired?"].fillna(False).astype(bool)
    degenerate = frame[["Open", "High", "Low", "Close"]].nunique(axis=1).le(1)
    jumps = frame["Close"].pct_change(fill_method=None).abs().gt(0.08)
    flagged = frame.index[repaired & degenerate & jumps]
    rows = []
    for dt in flagged:
        rows.append(
            {
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "close": float(frame.loc[dt, "Close"]),
            }
        )
    return rows


def audit_raw_series(ticker: str) -> dict[str, Any]:
    frame = load_raw_bars(ticker)
    if frame is None or len(frame) == 0:
        return {"ticker": ticker, "missing": True}

    repaired_count = int(frame["Repaired?"].fillna(False).sum()) if "Repaired?" in frame.columns else 0
    close = frame["Close"].astype(float)
    returns = close.pct_change(fill_method=None).dropna()
    top_spikes = [
        {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "abs_return": float(val)}
        for idx, val in returns.abs().sort_values(ascending=False).head(5).items()
    ]

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "start": pd.Timestamp(frame.index[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d"),
        "repaired_count": repaired_count,
        "degenerate_repaired_jump_dates": degenerate_repaired_jumps(frame),
        "top_raw_spikes": top_spikes,
    }


def audit_processed_dataset(name: str) -> dict[str, Any]:
    frame = load_processed_matrix(name)
    if frame is None or len(frame) == 0:
        return {"dataset": name, "missing": True}

    columns: dict[str, Any] = {}
    flagged_events: list[dict[str, Any]] = []

    for asset in frame.columns:
        series = frame[asset].astype(float)
        returns = series.pct_change(fill_method=None).dropna()
        top_spikes = [
            {"date": pd.Timestamp(idx).strftime("%Y-%m-%d"), "abs_return": float(val)}
            for idx, val in returns.abs().sort_values(ascending=False).head(5).items()
        ]
        threshold = spike_threshold(asset)
        suspicious = [
            {"date": row["date"], "abs_return": row["abs_return"]}
            for row in top_spikes
            if row["abs_return"] >= threshold
        ]
        if suspicious:
            flagged_events.append({"asset": asset, "events": suspicious})

        columns[asset] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "non_positive_count": int(series.le(0).sum()),
            "max_abs_return": float(returns.abs().max()) if len(returns) else 0.0,
            "p99_abs_return": float(returns.abs().quantile(0.99)) if len(returns) else 0.0,
            "max_stale_run": max_stale_run(series),
            "top_spikes": top_spikes,
        }

    notes: list[str] = []
    if name in {"india_research", "india_benchmark"}:
        notes.append("SILVER is currently a synthetic global silver in INR series unless config/data/silver_india_daily.csv is supplied.")
    if name == "india_tradable":
        notes.append("Tradable ETF history is shorter and should be used for live/paper workflows, not long-horizon benchmarking.")
    if name == "india_macro":
        notes.append("Macro series can legitimately exhibit larger jumps, especially INDIAVIX and US10Y in stress regimes.")

    raw_audit = {
        asset: audit_raw_series(ticker)
        for asset, ticker in DATASET_TICKERS.get(name, {}).items()
    }

    return {
        "dataset": name,
        "rows": int(len(frame)),
        "start": pd.Timestamp(frame.index[0]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d"),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "nan_cells": int(frame.isna().sum().sum()),
        "columns": columns,
        "flagged_events": flagged_events,
        "raw_sources": raw_audit,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local offline market datasets for obvious spikes and bad bars.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    payload = {
        "data_root": str(DATA_ROOT),
        "datasets": {name: audit_processed_dataset(name) for name in dataset_names()},
    }
    output_path = Path(args.output).expanduser().resolve()
    write_json(output_path, payload)

    print(f"Saved audit to {output_path}")
    for name, info in payload["datasets"].items():
        flagged = info.get("flagged_events", [])
        print(f"\n{name}: {info.get('start')} -> {info.get('end')} | rows {info.get('rows')} | flagged assets {len(flagged)}")
        for row in flagged[:5]:
            events = ", ".join(f"{event['date']}:{event['abs_return']:.2%}" for event in row["events"][:3])
            print(f"  {row['asset']:<10} {events}")
        raw_flags = []
        for asset, raw in info.get("raw_sources", {}).items():
            bad = raw.get("degenerate_repaired_jump_dates", [])
            if bad:
                raw_flags.append((asset, bad))
        for asset, bad in raw_flags[:5]:
            print(f"  raw:{asset:<6} degenerate repaired jump -> {bad}")


if __name__ == "__main__":
    main()
