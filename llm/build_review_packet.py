#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.alpha_v12_meta_ensemble import BASE_V9
from runtime.store import read_json, write_json
from strategy.v9_engine import YahooFinanceSource, load_llm_overlay, run_strategy


BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
PACKET_PATH = CACHE_DIR / "alpha_v12_latest_llm_packet.json"
ACTIVE_OVERLAY_PATH = CACHE_DIR / "active_overlay_latest.json"
PATTERN_LAB_PATH = CACHE_DIR / "pattern_signal_lab.json"
OUTPUT_PATH = CACHE_DIR / "llm_review_packet_latest.json"


def top_changes(base_weights: dict[str, float], final_weights: dict[str, float], top_k: int = 5) -> list[dict[str, float | str]]:
    rows = []
    for asset in sorted(set(base_weights) | set(final_weights)):
        base = float(base_weights.get(asset, 0.0))
        final = float(final_weights.get(asset, 0.0))
        rows.append(
            {
                "asset": asset,
                "base_weight": base,
                "final_weight": final,
                "delta_weight": final - base,
            }
        )
    rows = sorted(rows, key=lambda row: abs(float(row["delta_weight"])), reverse=True)
    return rows[:top_k]


def condense_pattern_context(pattern_lab: dict, changed_assets: list[str], top_k: int = 3) -> dict:
    condensed: dict[str, dict] = {}
    for dataset_name, dataset_info in (pattern_lab or {}).items():
        asset_map = dataset_info.get("assets", {}) if isinstance(dataset_info, dict) else {}
        dataset_payload = {}
        for asset in changed_assets:
            asset_info = asset_map.get(asset)
            if not isinstance(asset_info, dict):
                continue
            dataset_payload[asset] = {
                "top_positive": list(asset_info.get("top_positive", []))[:top_k],
                "top_negative": list(asset_info.get("top_negative", []))[:top_k],
            }
        if dataset_payload:
            condensed[dataset_name] = dataset_payload
    return condensed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review packet for an LLM critic/disagreement pass.")
    parser.add_argument("--start", default="2012-04-01")
    parser.add_argument("--packet-file", default=str(PACKET_PATH))
    parser.add_argument("--overlay-file", default=str(ACTIVE_OVERLAY_PATH))
    parser.add_argument("--pattern-file", default=str(PATTERN_LAB_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    prices = YahooFinanceSource(universe_mode="benchmark").fetch(args.start)
    base_weights_frame = run_strategy(prices, BASE_V9)
    overlay_path = Path(args.overlay_file).expanduser().resolve()
    packet_path = Path(args.packet_file).expanduser().resolve()
    pattern_path = Path(args.pattern_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    overlay = load_llm_overlay(str(overlay_path), prices.index) if overlay_path.exists() else None
    final_weights_frame = run_strategy(prices, BASE_V9, overlay=overlay)

    base_weights = {asset: float(base_weights_frame.iloc[-1][asset]) for asset in base_weights_frame.columns}
    final_weights = {asset: float(final_weights_frame.iloc[-1][asset]) for asset in final_weights_frame.columns}

    llm_packet = read_json(packet_path, default={})
    pattern_lab = read_json(pattern_path, default={})
    overlay_payload = read_json(overlay_path, default={})

    biggest_moves = top_changes(base_weights, final_weights)
    changed_assets = [str(row["asset"]) for row in biggest_moves if abs(float(row["delta_weight"])) > 1e-6]

    payload = {
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "base_quant_weights": base_weights,
        "overlay_adjusted_weights": final_weights,
        "largest_changes": biggest_moves,
        "macro_packet": llm_packet,
        "active_overlay": overlay_payload,
        "historical_pattern_context": condense_pattern_context(pattern_lab, changed_assets),
    }
    write_json(output_path, payload)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
