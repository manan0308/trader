#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from trader_system.runtime.env_loader import load_runtime_env
from trader_system.runtime.store import read_json, write_json


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "cache"
DEFAULT_PACKET = CACHE_DIR / "llm_review_packet_latest.json"
DEFAULT_PROMPT = BASE_DIR / "config" / "prompts" / "llm_disagreement_review_prompt.md"
DEFAULT_OUTPUT = CACHE_DIR / "llm_disagreement_review_latest.json"
CALL_CACHE_DIR = CACHE_DIR / "anthropic_review_calls"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-opus-4-6"


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


def call_paths(req_hash: str) -> tuple[Path, Path]:
    CALL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CALL_CACHE_DIR / f"{req_hash}.review.json", CALL_CACHE_DIR / f"{req_hash}.response.json"


def tool_schema() -> Dict[str, Any]:
    return {
        "name": "submit_review",
        "description": (
            "Return a strict review of whether the overlay-adjusted position is justified. "
            "Prefer agreeing with the quant baseline unless evidence is strong."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": ["agree", "partially_agree", "disagree"]},
                "confidence": {"type": "number"},
                "main_reason": {"type": "string"},
                "issue_type": {
                    "type": "string",
                    "enum": ["none", "size", "direction", "timing", "holding_period", "weak_evidence"],
                },
                "historical_support": {"type": "string"},
                "suggested_risk_adjustment": {"type": "number"},
                "suggested_holding_days": {"type": "integer"},
                "watch_items": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "verdict",
                "confidence",
                "main_reason",
                "issue_type",
                "historical_support",
                "suggested_risk_adjustment",
                "suggested_holding_days",
                "watch_items",
            ],
        },
    }


def anthropic_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def call_anthropic(api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = stable_json(payload).encode("utf-8")
    req = urllib.request.Request(ANTHROPIC_API_URL, data=body, headers=anthropic_headers(api_key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        content = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {content}") from exc


def extract_review(response: Dict[str, Any]) -> Dict[str, Any]:
    for block in response.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "submit_review":
            return dict(block.get("input", {}))
    raise RuntimeError(f"No submit_review tool call found. stop_reason={response.get('stop_reason')}")


def normalize_review(raw: Dict[str, Any]) -> Dict[str, Any]:
    verdict = str(raw.get("verdict", "partially_agree"))
    if verdict not in {"agree", "partially_agree", "disagree"}:
        verdict = "partially_agree"

    issue_type = str(raw.get("issue_type", "weak_evidence"))
    if issue_type not in {"none", "size", "direction", "timing", "holding_period", "weak_evidence"}:
        issue_type = "weak_evidence"

    confidence = float(np.clip(float(raw.get("confidence", 0.5)), 0.0, 1.0))
    suggested_risk_adjustment = float(np.clip(float(raw.get("suggested_risk_adjustment", 0.0)), -0.10, 0.10))
    suggested_holding_days = int(np.clip(int(raw.get("suggested_holding_days", 5)), 1, 10))
    main_reason = str(raw.get("main_reason", "")).strip()[:500]
    historical_support = str(raw.get("historical_support", "")).strip()[:200]
    watch_items = [str(item).strip()[:160] for item in (raw.get("watch_items") or []) if str(item).strip()][:5]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "main_reason": main_reason,
        "issue_type": issue_type,
        "historical_support": historical_support,
        "suggested_risk_adjustment": suggested_risk_adjustment,
        "suggested_holding_days": suggested_holding_days,
        "watch_items": watch_items,
    }


def build_payload(model: str, prompt_text: str, packet: Dict[str, Any], enable_web_search: bool, max_uses: int) -> Dict[str, Any]:
    tools: List[Dict[str, Any]] = []
    if enable_web_search:
        tools.append({"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses})
    tools.append(tool_schema())
    return {
        "model": model,
        "max_tokens": 700,
        "temperature": 0.0,
        "system": prompt_text,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Review the base quant position versus the overlay-adjusted position.\n"
                            "If web search is available, use it sparingly to check fresh macro or geopolitical context.\n"
                            "Do not restate the packet in prose. Finish by calling submit_review exactly once.\n\n"
                            f"{json.dumps(packet, indent=2)}"
                        ),
                    }
                ],
            }
        ],
        "tools": tools,
    }
    if enable_web_search:
        payload["tool_choice"] = {"type": "auto"}
    else:
        payload["tool_choice"] = {"type": "tool", "name": "submit_review"}
    return payload


def main() -> None:
    load_runtime_env(override=True)
    parser = argparse.ArgumentParser(description="Run an Anthropic disagreement review on the latest review packet.")
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--web-search", dest="web_search", action="store_true")
    parser.add_argument("--no-web-search", dest="web_search", action="store_false")
    parser.add_argument("--web-search-uses", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(web_search=False)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    packet_path = Path(args.packet).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    packet = read_json(packet_path, default={})
    prompt_text = prompt_path.read_text(encoding="utf-8")
    req_hash = request_hash(args.model, prompt_text, packet, args.web_search, args.web_search_uses)
    cached_review_path, cached_response_path = call_paths(req_hash)
    if cached_review_path.exists() and not args.force:
        cached = read_json(cached_review_path, default={})
        if isinstance(cached, dict):
            write_json(output_path, cached)
            print(f"Reused cached review: {cached_review_path}")
            print(json.dumps(cached.get("review", {}), indent=2))
            return

    response = call_anthropic(
        api_key,
        build_payload(args.model, prompt_text, packet, args.web_search, args.web_search_uses),
    )
    raw = extract_review(response)
    normalized = normalize_review(raw)
    payload = {
        "request_hash": req_hash,
        "model": args.model,
        "web_search": bool(args.web_search),
        "packet_path": str(packet_path),
        "review": normalized,
        "raw_response": response,
    }
    write_json(cached_review_path, payload)
    write_json(cached_response_path, response)
    write_json(output_path, payload)
    print(f"Saved to {output_path}")
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    main()
