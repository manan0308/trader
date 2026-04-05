from __future__ import annotations

import argparse
import html
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from trader_system.runtime.audit_log import stable_hash
from trader_system.runtime.store import read_json, read_jsonl, write_json, write_jsonl


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "events"
RAW_EVENTS_PATH = DATA_DIR / "raw" / "structured_events.jsonl"
ACTIVE_EVENTS_PATH = DATA_DIR / "derived" / "structured_event_store_active.json"
LATEST_EVENTS_PATH = BASE_DIR / "cache" / "structured_event_store_latest.json"

USER_AGENT = "Mozilla/5.0 (compatible; trader-system/1.0; +https://example.local)"

RSS_SOURCES: List[Dict[str, Any]] = [
    {
        "source_id": "rbi_press",
        "source": "RBI Press Releases",
        "url": "https://rbi.org.in/pressreleases_rss.xml",
        "category": "policy",
        "country": "IN",
        "official": True,
    },
    {
        "source_id": "rbi_notifications",
        "source": "RBI Notifications",
        "url": "https://rbi.org.in/notifications_rss.xml",
        "category": "policy",
        "country": "IN",
        "official": True,
    },
    {
        "source_id": "sebi_rss",
        "source": "SEBI RSS",
        "url": "https://www.sebi.gov.in/sebirss.xml",
        "category": "regulation",
        "country": "IN",
        "official": True,
    },
    {
        "source_id": "pib_economy",
        "source": "PIB Economy",
        "url": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",
        "category": "policy",
        "country": "IN",
        "official": True,
    },
    {
        "source_id": "mea_updates",
        "source": "MEA Updates",
        "url": "https://www.mea.gov.in/rss?SubMenuId=1&TagName=Latest%20Updates",
        "category": "geopolitics",
        "country": "IN",
        "official": True,
    },
]

GDELT_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "india_energy_geopolitics",
        "query": '(India OR Indian OR RBI OR rupee) AND (oil OR crude OR Hormuz OR Iran OR sanctions OR war OR blockade)',
        "category": "geopolitics",
        "country": "GLOBAL",
        "assets": ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"],
        "keywords": ["oil", "crude", "hormuz", "iran", "sanction", "war", "blockade"],
    },
    {
        "query_id": "global_macro_risk",
        "query": '("Federal Reserve" OR "US Treasury yields" OR tariffs OR recession OR inflation shock) AND (markets OR stocks OR bonds)',
        "category": "macro",
        "country": "GLOBAL",
        "assets": ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"],
        "keywords": ["federal reserve", "fed", "treasury", "yield", "tariff", "recession", "inflation"],
    },
    {
        "query_id": "india_market_policy",
        "query": '(India OR RBI OR SEBI OR budget) AND (market OR liquidity OR policy OR regulation OR fiscal)',
        "category": "policy",
        "country": "IN",
        "assets": ["NIFTY", "MIDCAP", "SMALLCAP", "US"],
        "keywords": ["budget", "liquidity", "policy", "regulation", "rbi", "sebi", "fii"],
    },
]

HIGH_SEVERITY_WORDS = {
    "war",
    "blockade",
    "attack",
    "missile",
    "sanction",
    "emergency",
    "default",
    "crisis",
    "tariff",
}
MEDIUM_SEVERITY_WORDS = {
    "oil",
    "crude",
    "inflation",
    "rate",
    "policy",
    "volatility",
    "budget",
    "election",
    "liquidity",
    "yield",
}
MARKET_RELEVANT_KEYWORDS = {
    "repo",
    "rate",
    "inflation",
    "liquidity",
    "forex",
    "foreign exchange",
    "inter-bank",
    "bond",
    "securities",
    "borrowing",
    "rupee",
    "oil",
    "crude",
    "budget",
    "fiscal",
    "regulation",
    "volatility",
    "market",
    "risk management",
    "sanction",
    "war",
    "tariff",
}
LOW_SIGNAL_PATTERNS = (
    "auction of state government securities",
    "indicative calendar of market borrowings",
    "weekly statistical supplement",
    "operational guidelines",
    "floating rate savings bonds",
    "state government securities",
)

ASSET_KEYWORDS = {
    "GOLD": {"gold", "bullion", "safe haven", "jewellery"},
    "SILVER": {"silver", "bullion", "mcx silver"},
    "US": {"nasdaq", "tech stocks", "us equities", "qqq", "mon100", "wall street", "federal reserve"},
    "NIFTY": {"nifty", "india equities", "indian stocks", "sensex", "fii", "dii"},
    "MIDCAP": {"midcap", "mid-cap", "broader markets"},
    "SMALLCAP": {"smallcap", "small-cap", "broader markets"},
}


@dataclass(frozen=True)
class EventStoreArtifacts:
    latest_path: Path = LATEST_EVENTS_PATH
    active_path: Path = ACTIVE_EVENTS_PATH
    raw_path: Path = RAW_EVENTS_PATH


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LATEST_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z") and "T" in text:
            return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    try:
        text = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fetch_url(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def text_from_html(raw: str) -> str:
    stripped = re.sub(r"<[^>]+>", " ", raw or "")
    stripped = html.unescape(stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def compact_text(raw: str, limit: int = 220) -> str:
    text = text_from_html(raw)
    return text[:limit].strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def contains_keywords(text: str, keywords: Iterable[str]) -> bool:
    tokens = token_set(text)
    lowered = text.lower()
    for keyword in keywords:
        key = keyword.lower()
        if " " in key:
            if key in lowered:
                return True
        elif key in tokens:
            return True
    return False


def infer_assets(title: str, summary: str, category: str, explicit_assets: Iterable[str] | None = None) -> List[str]:
    if explicit_assets:
        return list(dict.fromkeys(str(asset) for asset in explicit_assets))
    haystack = f"{title} {summary}".lower()
    assets: List[str] = []
    for asset, keywords in ASSET_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            assets.append(asset)
    if category in {"policy", "macro"}:
        assets.extend(["NIFTY", "MIDCAP", "SMALLCAP", "US"])
    if "oil" in haystack or "crude" in haystack or "war" in haystack:
        assets.extend(["GOLD", "SILVER"])
    return list(dict.fromkeys(asset for asset in assets if asset))


def severity_from_event(title: str, summary: str, *, official: bool, category: str) -> float:
    haystack = f"{title} {summary}".lower()
    severity = 0.35
    if contains_keywords(haystack, HIGH_SEVERITY_WORDS):
        severity = 0.85
    elif contains_keywords(haystack, MEDIUM_SEVERITY_WORDS):
        severity = 0.60
    if official:
        severity += 0.05
    if category == "geopolitics":
        severity += 0.05
    return float(min(severity, 0.95))


def horizon_days_for_event(title: str, summary: str, category: str) -> int:
    haystack = f"{title} {summary}".lower()
    if any(word in haystack for word in {"war", "blockade", "attack", "emergency", "crisis"}):
        return 5
    if any(word in haystack for word in {"budget", "policy", "regulation", "rate", "tariff"}):
        return 7
    if category == "macro":
        return 5
    return 3


def build_event(
    *,
    source_id: str,
    source: str,
    source_type: str,
    source_url: str,
    title: str,
    summary: str,
    published_at: datetime,
    category: str,
    country: str,
    official: bool,
    assets: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
) -> Dict[str, Any]:
    assets_list = infer_assets(title, summary, category, explicit_assets=assets)
    severity = severity_from_event(title, summary, official=official, category=category)
    horizon_days = horizon_days_for_event(title, summary, category)
    expires_at = (published_at + timedelta(days=horizon_days)).astimezone(timezone.utc)
    event_key = stable_hash(
        {
            "source_id": source_id,
            "title": title,
            "source_url": source_url,
            "published_at": published_at.isoformat(),
        }
    )[:16]
    return {
        "event_id": event_key,
        "source_id": source_id,
        "source": source,
        "source_type": source_type,
        "source_url": source_url,
        "title": title[:220],
        "summary": compact_text(summary),
        "published_at": published_at.astimezone(timezone.utc).isoformat(),
        "category": category,
        "country": country,
        "official": bool(official),
        "severity": severity,
        "horizon_days": horizon_days,
        "expires_at": expires_at.isoformat(),
        "assets": assets_list,
        "tags": list(dict.fromkeys(str(tag) for tag in (tags or []))),
    }


def is_low_signal_event(title: str, summary: str) -> bool:
    haystack = f"{title} {summary}".lower()
    return any(pattern in haystack for pattern in LOW_SIGNAL_PATTERNS)


def is_market_relevant_event(title: str, summary: str) -> bool:
    return contains_keywords(f"{title} {summary}".lower(), MARKET_RELEVANT_KEYWORDS)


def fetch_rss_source(spec: Dict[str, Any], *, max_items: int = 12) -> List[Dict[str, Any]]:
    payload = fetch_url(spec["url"])
    root = ET.fromstring(payload)
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{*}entry")
    events: List[Dict[str, Any]] = []
    for item in items[:max_items]:
        title = (
            item.findtext("title")
            or item.findtext("{*}title")
            or ""
        ).strip()
        link = item.findtext("link") or item.findtext("{*}link") or spec["url"]
        if not link and item.find("{*}link") is not None:
            link = item.find("{*}link").attrib.get("href", spec["url"])
        summary = (
            item.findtext("description")
            or item.findtext("{*}summary")
            or item.findtext("{*}content")
            or ""
        )
        published_at = parse_timestamp(
            item.findtext("pubDate")
            or item.findtext("{*}published")
            or item.findtext("{*}updated")
        )
        if not title or published_at is None:
            continue
        events.append(
            build_event(
                source_id=spec["source_id"],
                source=spec["source"],
                source_type="rss",
                source_url=link,
                title=title,
                summary=summary,
                published_at=published_at,
                category=spec["category"],
                country=spec["country"],
                official=bool(spec["official"]),
            )
        )
    return events


def fetch_gdelt_query(spec: Dict[str, Any], *, max_records: int = 4) -> List[Dict[str, Any]]:
    params = {
        "query": spec["query"],
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "sort": "DateDesc",
        "timespan": "7d",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    payload = json.loads(fetch_url(url).decode("utf-8"))
    events: List[Dict[str, Any]] = []
    for article in payload.get("articles", []) or []:
        published_at = parse_timestamp(article.get("seendate"))
        title = str(article.get("title", "")).strip()
        if not title or published_at is None:
            continue
        if not contains_keywords(title.lower(), spec.get("keywords", [])):
            continue
        summary = f"{article.get('domain', '')} | {article.get('language', '')} | {article.get('sourcecountry', '')}".strip(" |")
        events.append(
            build_event(
                source_id=spec["query_id"],
                source=f"GDELT:{spec['query_id']}",
                source_type="gdelt",
                source_url=str(article.get("url", "")),
                title=title,
                summary=summary,
                published_at=published_at,
                category=spec["category"],
                country=spec["country"],
                official=False,
                assets=spec.get("assets"),
                tags=[spec["query_id"]],
            )
        )
    return events


def dedupe_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for event in events:
        key = stable_hash(
            {
                "title": event.get("title"),
                "source_url": event.get("source_url"),
                "published_at": event.get("published_at"),
            }
        )[:16]
        previous = latest.get(key)
        if previous is None or float(event.get("severity", 0.0)) >= float(previous.get("severity", 0.0)):
            latest[key] = dict(event)
    return list(latest.values())


def active_events(events: Iterable[Dict[str, Any]], as_of: datetime) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in events:
        if str(event.get("source_type", "")).endswith("_error"):
            continue
        if bool(event.get("official")) and not is_market_relevant_event(str(event.get("title", "")), str(event.get("summary", ""))):
            continue
        published_at = parse_timestamp(event.get("published_at"))
        expires_at = parse_timestamp(event.get("expires_at"))
        if published_at is None or expires_at is None:
            continue
        if published_at <= as_of and expires_at >= as_of:
            if is_low_signal_event(str(event.get("title", "")), str(event.get("summary", ""))) and float(event.get("severity", 0.0)) < 0.70:
                continue
            rows.append(dict(event))
    rows.sort(
        key=lambda row: (
            float(row.get("severity", 0.0)),
            str(row.get("published_at", "")),
        ),
        reverse=True,
    )
    return rows


def summarize_events(events: List[Dict[str, Any]], *, max_events: int = 8) -> Dict[str, Any]:
    category_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    asset_watch: Dict[str, List[str]] = {}
    for event in events:
        category = str(event.get("category", "other"))
        source = str(event.get("source", "unknown"))
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        for asset in event.get("assets", []) or []:
            asset_watch.setdefault(asset, [])
            if len(asset_watch[asset]) < 3:
                asset_watch[asset].append(str(event.get("title", "")))

    top_events = [
        {
            "event_id": event["event_id"],
            "date": str(event["published_at"])[:10],
            "source": event["source"],
            "category": event["category"],
            "severity": event["severity"],
            "assets": event["assets"],
            "title": event["title"],
            "summary": event["summary"],
            "source_url": event["source_url"],
        }
        for event in events[:max_events]
    ]

    return {
        "event_count": len(events),
        "category_counts": category_counts,
        "source_counts": source_counts,
        "asset_watch": asset_watch,
        "top_events": top_events,
    }


def build_historical_signal_hints(pattern_lab: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    benchmark_assets = (pattern_lab.get("india_benchmark") or {}).get("assets", {})
    if benchmark_assets:
        hints.append(
            {
                "pattern": "equity_rebound_after_drawdown",
                "assets": ["NIFTY", "MIDCAP", "SMALLCAP", "US"],
                "note": "Deep 20-day drawdowns in broader-risk sleeves often mean-reverted over the next 63 days, especially when the long-term trend was intact.",
            }
        )
        hints.append(
            {
                "pattern": "gold_after_vol_spike",
                "assets": ["GOLD"],
                "note": "Gold historically held up better after volatility spikes than during calm periods.",
            }
        )
        hints.append(
            {
                "pattern": "silver_breakout_persistence",
                "assets": ["SILVER"],
                "note": "Silver breakouts near 252-day highs showed positive medium-horizon follow-through more often than not.",
            }
        )
    return hints


def refresh_structured_event_store(
    *,
    as_of: str | None = None,
    refresh: bool = False,
    rss_limit: int = 10,
    gdelt_limit: int = 4,
) -> Dict[str, Any]:
    ensure_dirs()
    as_of_dt = parse_timestamp(as_of) if as_of else now_utc()
    if as_of_dt is None:
        as_of_dt = now_utc()
    as_of_dt = as_of_dt.astimezone(timezone.utc)
    as_of_key = as_of_dt.strftime("%Y-%m-%d")

    cached = read_json(LATEST_EVENTS_PATH, default=None)
    if (
        not refresh
        and isinstance(cached, dict)
        and str(cached.get("as_of")) == as_of_key
        and isinstance(cached.get("events"), list)
    ):
        return cached

    fetched: List[Dict[str, Any]] = []
    for spec in RSS_SOURCES:
        try:
            fetched.extend(fetch_rss_source(spec, max_items=rss_limit))
        except Exception as exc:
            fetched.append(
                {
                    "event_id": stable_hash({"source": spec["source_id"], "error": str(exc)})[:16],
                    "source_id": spec["source_id"],
                    "source": spec["source"],
                    "source_type": "rss_error",
                    "source_url": spec["url"],
                    "title": f"{spec['source']} fetch error",
                    "summary": str(exc)[:220],
                    "published_at": as_of_dt.isoformat(),
                    "category": spec["category"],
                    "country": spec["country"],
                    "official": bool(spec["official"]),
                    "severity": 0.10,
                    "horizon_days": 1,
                    "expires_at": (as_of_dt + timedelta(days=1)).isoformat(),
                    "assets": [],
                    "tags": ["fetch_error"],
                }
            )

    for spec in GDELT_QUERIES:
        try:
            fetched.extend(fetch_gdelt_query(spec, max_records=gdelt_limit))
        except Exception:
            continue

    deduped = dedupe_events(fetched)
    deduped.sort(
        key=lambda row: (
            float(row.get("severity", 0.0)),
            str(row.get("published_at", "")),
        ),
        reverse=True,
    )
    active = active_events(deduped, as_of_dt)
    summary = summarize_events(active)

    history = read_jsonl(RAW_EVENTS_PATH)
    history_map = {str(row.get("event_id")): row for row in history}
    for row in deduped:
        history_map[str(row.get("event_id"))] = row
    write_jsonl(RAW_EVENTS_PATH, list(history_map.values())[-500:])

    payload = {
        "as_of": as_of_key,
        "generated_at": now_utc().isoformat(),
        "events": active,
        "summary": summary,
    }
    write_json(ACTIVE_EVENTS_PATH, payload)
    write_json(LATEST_EVENTS_PATH, payload)
    return payload


def attach_structured_context(
    packet: Dict[str, Any],
    *,
    event_store: Dict[str, Any] | None = None,
    pattern_lab: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    enriched = dict(packet)
    if isinstance(event_store, dict):
        enriched["structured_event_summary"] = event_store.get("summary", {})
        enriched["structured_event_facts"] = (event_store.get("summary", {}) or {}).get("top_events", [])
        enriched["structured_event_store_path"] = str(LATEST_EVENTS_PATH)
    if isinstance(pattern_lab, dict):
        enriched["historical_signal_hints"] = build_historical_signal_hints(pattern_lab)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the structured event store used by the LLM overlay.")
    parser.add_argument("--as-of", help="UTC or local date/time to anchor the event store.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--rss-limit", type=int, default=10)
    parser.add_argument("--gdelt-limit", type=int, default=4)
    args = parser.parse_args()

    payload = refresh_structured_event_store(
        as_of=args.as_of,
        refresh=args.refresh,
        rss_limit=args.rss_limit,
        gdelt_limit=args.gdelt_limit,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"\nSaved to {LATEST_EVENTS_PATH}")


if __name__ == "__main__":
    main()
