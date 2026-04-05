#!/usr/bin/env python3
"""
Pattern Signal Lab
==================

Mine simple conditional patterns from saved offline history:
- RSI extremes
- trend / breakout
- drawdown stress
- volatility spike
- volume surge (when volume exists)

The goal is not to auto-trade these directly, but to discover obvious tells
that can later feed either the quant model or an LLM review layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_data.market_store import (
    INDIA_BENCHMARK_TICKERS,
    INDIA_RESEARCH_TICKERS,
    INDIA_TRADABLE_TICKERS,
    US_ANALOG_UNIVERSES,
    load_processed_matrix,
    load_raw_bars,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = BASE_DIR / "cache" / "pattern_signal_lab.json"
HORIZONS = [5, 20, 63]

DATASET_TICKERS: Dict[str, Dict[str, str]] = {
    "india_research": INDIA_RESEARCH_TICKERS,
    "india_benchmark": INDIA_BENCHMARK_TICKERS,
    "india_tradable": INDIA_TRADABLE_TICKERS,
    "us_analog_shy": US_ANALOG_UNIVERSES["us_analog_shy"],
    "us_analog_ief": US_ANALOG_UNIVERSES["us_analog_ief"],
    "us_analog_bil": US_ANALOG_UNIVERSES["us_analog_bil"],
}


def rsi(series: pd.Series, lookback: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.rolling(lookback).mean() / down.rolling(lookback).mean().replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-horizon) / series - 1.0


def load_volume_series(ticker: str) -> pd.Series | None:
    frame = load_raw_bars(ticker)
    if frame is None:
        return None
    if "Volume" not in frame.columns:
        return None
    vol = pd.Series(frame["Volume"].astype(float).values, index=pd.to_datetime(frame.index))
    if vol.sum() <= 0:
        return None
    return vol.sort_index()


def pattern_events(close: pd.Series, volume: pd.Series | None = None) -> Dict[str, pd.Series]:
    ret = close.pct_change(fill_method=None)
    sma200 = close.rolling(200).mean()
    rsi14 = rsi(close, 14)
    dd20 = close / close.rolling(20).max() - 1.0
    vol_ratio = ret.rolling(20).std() / ret.rolling(252).std()
    breakout = close >= close.rolling(252).max().shift(1)

    events = {
        "trend_up": close > sma200,
        "trend_down": close < sma200,
        "rsi_oversold": rsi14 < 30,
        "rsi_overbought": rsi14 > 70,
        "dd20_crash": dd20 < -0.10,
        "vol_spike": vol_ratio > 1.5,
        "breakout_252": breakout.fillna(False),
        "trend_up_rsi_reset": (close > sma200) & (rsi14 < 40),
        "downtrend_overbought": (close < sma200) & (rsi14 > 60),
    }

    if volume is not None:
        vol = volume.reindex(close.index).fillna(0.0)
        vol_z = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
        events["volume_surge"] = vol_z > 2.0

    return {name: series.fillna(False) for name, series in events.items()}


def summarize_event(close: pd.Series, event: pd.Series, horizons: Iterable[int]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for horizon in horizons:
        fwd = forward_return(close, horizon)
        sample = fwd[event].dropna()
        if len(sample) == 0:
            continue
        out[f"{horizon}d"] = {
            "count": int(len(sample)),
            "avg_forward_return": float(sample.mean()),
            "median_forward_return": float(sample.median()),
            "hit_rate": float((sample > 0).mean()),
        }
    return out


def analyze_dataset(name: str) -> Dict[str, object]:
    prices = load_processed_matrix(name)
    if prices is None or len(prices) == 0:
        raise RuntimeError(f"missing processed dataset: {name}")

    mapping = DATASET_TICKERS.get(name, {})
    asset_rows: Dict[str, object] = {}

    for asset in prices.columns:
        if asset == "CASH":
            continue
        close = prices[asset].dropna()
        ticker = mapping.get(asset)
        volume = load_volume_series(ticker) if ticker else None
        events = pattern_events(close, volume=volume)
        summaries = {event_name: summarize_event(close, event_mask, HORIZONS) for event_name, event_mask in events.items()}

        scored: List[tuple[str, str, Dict[str, float]]] = []
        for event_name, horizon_map in summaries.items():
            for horizon_key, stats in horizon_map.items():
                if stats["count"] < 20:
                    continue
                scored.append((event_name, horizon_key, stats))

        positive = sorted(scored, key=lambda row: (row[2]["avg_forward_return"], row[2]["hit_rate"]), reverse=True)[:8]
        negative = sorted(scored, key=lambda row: (row[2]["avg_forward_return"], row[2]["hit_rate"]))[:8]

        asset_rows[asset] = {
            "top_positive": [
                {"event": event_name, "horizon": horizon_key, **stats}
                for event_name, horizon_key, stats in positive
            ],
            "top_negative": [
                {"event": event_name, "horizon": horizon_key, **stats}
                for event_name, horizon_key, stats in negative
            ],
        }

    return {
        "dataset": name,
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "rows": len(prices),
        "assets": asset_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine simple historical pattern tells from offline datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_TICKERS.keys()),
        help="Dataset(s) to analyze. Defaults to india_benchmark and us_analog_shy.",
    )
    args = parser.parse_args()

    selected = args.dataset or ["india_benchmark", "us_analog_shy"]
    payload = {name: analyze_dataset(name) for name in selected}
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for name in selected:
        info = payload[name]
        print(f"{name}: {info['sample_start']} -> {info['sample_end']} | rows {info['rows']}")
        for asset, summary in info["assets"].items():
            best = summary["top_positive"][0] if summary["top_positive"] else None
            worst = summary["top_negative"][0] if summary["top_negative"] else None
            if best:
                print(f"  {asset:<10} best {best['event']} {best['horizon']} avg {best['avg_forward_return']:.2%} hit {best['hit_rate']:.0%}")
            if worst:
                print(f"  {asset:<10} worst {worst['event']} {worst['horizon']} avg {worst['avg_forward_return']:.2%} hit {worst['hit_rate']:.0%}")
    print(f"\nSaved to {CACHE_PATH}")


if __name__ == "__main__":
    main()
