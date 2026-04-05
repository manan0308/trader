#!/usr/bin/env python3
"""
Anthropic Overlay Runner
========================

Generate a validated LLM macro override from the latest v12 packet using the
Anthropic Messages API, then optionally rerun the strategy with that override.

Design priorities:
- Use Anthropic only.
- Keep Opus calls sparse and cacheable.
- Force structured output through a client tool schema.
- Validate and clip overrides locally before applying them.
- Let the quant model remain the allocator of record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trader_system.strategy.v9_engine import (
    ALL,
    RISKY,
    DEFAULT_TX,
    NarrativeOverlay,
    YahooFinanceSource,
    load_llm_overlay,
    run_strategy,
)
from research.alpha_v11_macro_value_research import fetch_macro_panel
from research.alpha_v12_meta_ensemble import BASE_V9, meta_candidates, run_meta_strategy
from research.alpha_v13_sparse_meta import risk_preserving_candidates, run_sparse_meta_strategy
from research.alpha_v14_sparse_sleeves import run_sparse_sleeve_strategy, sleeve_sparse_candidates
from trader_system.runtime.env_loader import load_runtime_env


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "cache"
DEFAULT_PACKET = CACHE_DIR / "alpha_v12_latest_llm_packet.json"
DEFAULT_PROMPT = BASE_DIR / "config" / "prompts" / "llm_overlay_meta_prompt.md"
DEFAULT_OUTPUT = CACHE_DIR / "anthropic_overlay_latest.json"
CALL_CACHE_DIR = CACHE_DIR / "anthropic_overlay_calls"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-6"


@dataclass(frozen=True)
class CallArtifacts:
    request_hash: str
    output_path: Path
    raw_response_path: Path
    meta_path: Path


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def request_hash(model: str, prompt_text: str, packet: Dict[str, Any], enable_web_search: bool, max_uses: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt_text,
        "packet": packet,
        "enable_web_search": enable_web_search,
        "max_uses": max_uses,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def call_paths(req_hash: str) -> CallArtifacts:
    CALL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CallArtifacts(
        request_hash=req_hash,
        output_path=CALL_CACHE_DIR / f"{req_hash}.overlay.json",
        raw_response_path=CALL_CACHE_DIR / f"{req_hash}.response.json",
        meta_path=CALL_CACHE_DIR / f"{req_hash}.meta.json",
    )


def overlay_tool_schema() -> Dict[str, Any]:
    bias_props = {asset: {"type": "number"} for asset in RISKY}
    return {
        "name": "submit_overlay",
        "description": (
            "Return the final validated macro risk override for the allocator. "
            "Always call this exactly once. Use low values by default, prefer short-lived overrides, "
            "and do not increase total portfolio risk beyond the base model."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "default_risk_off_override": {"type": "number"},
                "dates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "date": {"type": "string"},
                            "holding_days": {"type": "integer"},
                            "risk_off_override": {"type": "number"},
                            "asset_bias": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": bias_props,
                            },
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["date", "holding_days", "risk_off_override", "asset_bias", "rationale", "confidence"],
                    },
                },
            },
            "required": ["default_risk_off_override", "dates"],
        },
    }


def packet_event_risk(packet: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    macro_state = packet.get("macro_state", {})
    breadth = float(macro_state.get("breadth_above_200d", 1.0))
    vix_ratio = float(macro_state.get("indiavix_ratio_to_6m_median", 1.0))
    usd_inr = float(macro_state.get("usd_inr_3m_return", 0.0))
    crude = float(macro_state.get("crude_3m_return", 0.0))
    us10y = float(macro_state.get("us10y_3m_change_bps_proxy", 0.0))

    triggered = (
        vix_ratio > 1.20
        or breadth < 0.45
        or usd_inr > 0.04
        or crude > 0.20
        or us10y > 8.0
    )
    return triggered, {
        "breadth_above_200d": breadth,
        "indiavix_ratio_to_6m_median": vix_ratio,
        "usd_inr_3m_return": usd_inr,
        "crude_3m_return": crude,
        "us10y_3m_change_bps_proxy": us10y,
    }


def build_messages(packet: Dict[str, Any], enable_web_search: bool) -> List[Dict[str, Any]]:
    user_text = (
        "Here is the current allocator packet as JSON.\n\n"
        "Decide only whether a temporary defensive override is warranted.\n"
        "If web_search is available, check current macro, policy, geopolitical, and market-risk developments relevant to this packet before deciding.\n"
        "Keep searches sparse and focused.\n"
        "Finish by calling submit_overlay exactly once.\n\n"
        f"{json.dumps(packet, indent=2)}"
    )
    return [{"role": "user", "content": [{"type": "text", "text": user_text}]}]


def anthropic_request_payload(
    model: str,
    prompt_text: str,
    packet: Dict[str, Any],
    enable_web_search: bool,
    web_search_uses: int,
) -> Dict[str, Any]:
    tools: List[Dict[str, Any]] = []
    if enable_web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": web_search_uses})
    tools.append(overlay_tool_schema())

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": 900,
        "temperature": 0.0,
        "system": prompt_text,
        "messages": build_messages(packet, enable_web_search=enable_web_search),
        "tools": tools,
    }

    if enable_web_search:
        payload["tool_choice"] = {"type": "auto"}
    else:
        payload["tool_choice"] = {"type": "tool", "name": "submit_overlay"}
    return payload


def anthropic_finalize_payload(
    model: str,
    prompt_text: str,
    packet: Dict[str, Any],
    prior_response: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 400,
        "temperature": 0.0,
        "system": prompt_text,
        "messages": build_messages(packet, enable_web_search=True)
        + [
            {"role": "assistant", "content": prior_response.get("content", [])},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Now call submit_overlay exactly once with the final structured override. Do not add any prose.",
                    }
                ],
            },
        ],
        "tools": [overlay_tool_schema()],
        "tool_choice": {"type": "tool", "name": "submit_overlay"},
    }


def anthropic_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def call_anthropic(
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    body = stable_json(payload).encode("utf-8")
    req = urllib.request.Request(ANTHROPIC_API_URL, data=body, headers=anthropic_headers(api_key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        content = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {content}") from exc


def extract_overlay_from_response(response: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    content = response.get("content", [])
    for block in content:
        if block.get("type") == "tool_use" and block.get("name") == "submit_overlay":
            return block.get("input", {}), block

    text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text").strip()
    if text:
        try:
            return json.loads(text), {"type": "text"}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Anthropic returned text but not valid JSON: {text[:500]}") from exc

    raise RuntimeError(f"No submit_overlay tool_use found. stop_reason={response.get('stop_reason')}")


def normalize_overlay(raw: Dict[str, Any], packet: Dict[str, Any]) -> Dict[str, Any]:
    as_of = pd.Timestamp(packet["as_of"])

    def clip_bias(value: Any) -> float:
        try:
            return float(np.clip(float(value), -1.0, 1.0))
        except Exception:
            return 0.0

    default_risk = float(np.clip(float(raw.get("default_risk_off_override", 0.0)), 0.0, 0.6))
    rows = []
    for item in raw.get("dates", []) or []:
        try:
            date = pd.Timestamp(item["date"])
        except Exception:
            continue
        if date < as_of:
            continue

        confidence = float(np.clip(float(item.get("confidence", 0.0)), 0.0, 1.0))
        risk = float(np.clip(float(item.get("risk_off_override", 0.0)), 0.0, 0.6))
        holding_days = int(np.clip(int(item.get("holding_days", 1)), 1, 20))
        rationale = str(item.get("rationale", "")).strip()[:500]

        bias_map = item.get("asset_bias", {}) if isinstance(item.get("asset_bias", {}), dict) else {}
        normalized_bias = {asset: clip_bias(bias_map.get(asset, 0.0)) for asset in RISKY}

        # Low-confidence overrides should not move the portfolio much.
        if confidence < 0.35:
            risk = min(risk, 0.10)
            normalized_bias = {asset: 0.0 for asset in RISKY}

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "holding_days": holding_days,
                "risk_off_override": risk,
                "asset_bias": normalized_bias,
                "rationale": rationale,
                "confidence": confidence,
            }
        )

    rows = sorted(rows, key=lambda item: item["date"])
    return {
        "default_risk_off_override": default_risk,
        "dates": rows,
    }


def save_call_artifacts(
    artifacts: CallArtifacts,
    overlay: Dict[str, Any],
    response: Dict[str, Any],
    meta: Dict[str, Any],
    output_path: Path,
) -> None:
    artifacts.output_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    artifacts.raw_response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    artifacts.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    output_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")


def rerun_latest_allocations(override_path: Path) -> Dict[str, Dict[str, float]]:
    prices = YahooFinanceSource().fetch("2012-01-01")
    overlay = load_llm_overlay(str(override_path), prices.index)
    macro = fetch_macro_panel(prices.index[0].strftime("%Y-%m-%d"))

    v9_base = run_strategy(prices, BASE_V9)
    v9_llm = run_strategy(prices, BASE_V9, overlay=overlay)

    v12_cfg = meta_candidates()[3]
    v12_base = run_meta_strategy(prices, macro, v12_cfg, tx_cost=DEFAULT_TX)
    v12_llm = run_meta_strategy(prices, macro, v12_cfg, overlay=overlay, tx_cost=DEFAULT_TX)

    v13_cfg = risk_preserving_candidates()[0]
    v13_base = run_sparse_meta_strategy(prices, macro, v13_cfg, tx_cost=DEFAULT_TX)
    v13_llm = run_sparse_meta_strategy(prices, macro, v13_cfg, overlay=overlay, tx_cost=DEFAULT_TX)

    v14_cfg = sleeve_sparse_candidates()[0]
    v14_base = run_sparse_sleeve_strategy(prices, macro, v14_cfg, tx_cost=DEFAULT_TX)
    v14_llm = run_sparse_sleeve_strategy(prices, macro, v14_cfg, overlay=overlay, tx_cost=DEFAULT_TX)

    def latest_diff(base: pd.DataFrame, llm: pd.DataFrame) -> Dict[str, float]:
        delta = (llm.iloc[-1] - base.iloc[-1]).reindex(ALL).fillna(0.0)
        return {asset: float(delta[asset]) for asset in ALL if abs(float(delta[asset])) >= 0.001}

    return {
        "v9_base": {asset: float(v9_base.iloc[-1][asset]) for asset in ALL},
        "v9_llm": {asset: float(v9_llm.iloc[-1][asset]) for asset in ALL},
        "v9_delta": latest_diff(v9_base, v9_llm),
        "v12_base": {asset: float(v12_base.iloc[-1][asset]) for asset in ALL},
        "v12_llm": {asset: float(v12_llm.iloc[-1][asset]) for asset in ALL},
        "v12_delta": latest_diff(v12_base, v12_llm),
        "v13_base": {asset: float(v13_base.iloc[-1][asset]) for asset in ALL},
        "v13_llm": {asset: float(v13_llm.iloc[-1][asset]) for asset in ALL},
        "v13_delta": latest_diff(v13_base, v13_llm),
        "v14_base": {asset: float(v14_base.iloc[-1][asset]) for asset in ALL},
        "v14_llm": {asset: float(v14_llm.iloc[-1][asset]) for asset in ALL},
        "v14_delta": latest_diff(v14_base, v14_llm),
    }


def main() -> None:
    load_runtime_env(override=True)
    parser = argparse.ArgumentParser(description="Call Anthropic to generate a validated macro overlay JSON.")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--enable-web-search", dest="enable_web_search", action="store_true")
    parser.add_argument("--disable-web-search", dest="enable_web_search", action="store_false")
    parser.add_argument("--web-search-uses", type=int, default=2)
    parser.add_argument("--call-mode", choices=["event", "always"], default="always")
    parser.add_argument("--force", action="store_true", help="Ignore local call cache.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-strategy", action="store_true")
    parser.set_defaults(enable_web_search=True)
    args = parser.parse_args()

    packet_path = Path(args.packet).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    packet = load_json(packet_path)
    prompt_text = load_text(prompt_path)
    event_trigger, risk_snapshot = packet_event_risk(packet)
    req_hash = request_hash(
        model=args.model,
        prompt_text=prompt_text,
        packet=packet,
        enable_web_search=args.enable_web_search,
        max_uses=args.web_search_uses,
    )
    artifacts = call_paths(req_hash)

    if artifacts.output_path.exists() and not args.force:
        overlay = load_json(artifacts.output_path)
        output_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        print(f"Reused cached Anthropic overlay: {artifacts.output_path}")
        print(f"Wrote latest overlay to: {output_path}")
        if args.rerun_strategy:
            print(json.dumps(rerun_latest_allocations(output_path), indent=2))
        return

    if args.call_mode == "event" and not event_trigger and not args.force:
        overlay = {"default_risk_off_override": 0.0, "dates": []}
        output_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        artifacts.output_path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
        artifacts.meta_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "packet": str(packet_path),
                    "prompt": str(prompt_path),
                    "enable_web_search": args.enable_web_search,
                    "web_search_uses": args.web_search_uses,
                    "call_mode": args.call_mode,
                    "skipped_api_call": True,
                    "risk_snapshot": risk_snapshot,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Skipped Anthropic API call because packet did not trip the event-risk pre-screen.")
        print(json.dumps({"risk_snapshot": risk_snapshot, "overlay": overlay}, indent=2))
        if args.rerun_strategy:
            print(json.dumps(rerun_latest_allocations(output_path), indent=2))
        return

    payload = anthropic_request_payload(
        model=args.model,
        prompt_text=prompt_text,
        packet=packet,
        enable_web_search=args.enable_web_search,
        web_search_uses=args.web_search_uses,
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

    response = call_anthropic(api_key=api_key, payload=payload)
    try:
        raw_overlay, evidence = extract_overlay_from_response(response)
    except RuntimeError:
        finalize_payload = anthropic_finalize_payload(
            model=args.model,
            prompt_text=prompt_text,
            packet=packet,
            prior_response=response,
        )
        response = call_anthropic(api_key=api_key, payload=finalize_payload)
        raw_overlay, evidence = extract_overlay_from_response(response)
    overlay = normalize_overlay(raw_overlay, packet=packet)

    meta = {
        "model": args.model,
        "packet": str(packet_path),
        "prompt": str(prompt_path),
        "enable_web_search": args.enable_web_search,
        "web_search_uses": args.web_search_uses,
        "usage": response.get("usage", {}),
        "stop_reason": response.get("stop_reason"),
        "evidence_block_type": evidence.get("type"),
    }
    save_call_artifacts(artifacts, overlay, response, meta, output_path=output_path)

    print(f"Anthropic overlay written to: {output_path}")
    print(json.dumps(overlay, indent=2))
    print("\nUsage metadata:")
    print(json.dumps(meta, indent=2))

    if args.rerun_strategy:
        print("\nLatest allocation delta with overlay:")
        print(json.dumps(rerun_latest_allocations(output_path), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
