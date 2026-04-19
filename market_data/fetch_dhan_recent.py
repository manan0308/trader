#!/usr/bin/env python3
"""
Fetch recent daily Dhan candles for the live ETF universe.

This is a pragmatic fallback when Groww recent-history cache is missing.
It writes both:
- a long-form daily file compatible with the repo's Dhan parquet layout
- an aligned close matrix for quick inspection / ad hoc backtests
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from dhanhq import dhanhq
from dhanhq.dhan_context import DhanContext


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "cache"


@dataclass(frozen=True)
class DhanInstrument:
    symbol: str
    security_id: str
    exchange_segment: str = dhanhq.NSE
    instrument: str = "EQUITY"


DEFAULT_UNIVERSE: Dict[str, DhanInstrument] = {
    "NIFTYBEES": DhanInstrument("NIFTYBEES", "10576"),
    "MID150BEES": DhanInstrument("MID150BEES", "8506"),
    "HDFCSML250": DhanInstrument("HDFCSML250", "14233"),
    "GOLDBEES": DhanInstrument("GOLDBEES", "14428"),
    "SILVERBEES": DhanInstrument("SILVERBEES", "8080"),
    "MON100": DhanInstrument("MON100", "22739"),
    "LIQUIDBEES": DhanInstrument("LIQUIDBEES", "11006"),
}


def decode_client_id(access_token: str) -> Optional[str]:
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        parsed = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
        return str(parsed.get("dhanClientId") or "")
    except Exception:
        return None


def epoch_to_ist_date(epoch_values: Iterable[float]) -> pd.Series:
    return (
        pd.to_datetime(pd.Series(list(epoch_values), dtype="float64"), unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )


def fetch_symbol_history(
    client: dhanhq,
    instrument: DhanInstrument,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    payload = None
    for attempt in range(5):
        payload = client.historical_daily_data(
            security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment,
            instrument_type=instrument.instrument,
            from_date=start_date,
            to_date=end_date,
        )
        if str(payload.get("status")) == "success":
            break
        remarks = payload.get("remarks") or {}
        if isinstance(remarks, dict) and str(remarks.get("error_code")) == "DH-904" and attempt < 4:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"{instrument.symbol}: {remarks}")

    data = payload.get("data") or {}
    if not data or not data.get("timestamp"):
        raise RuntimeError(f"{instrument.symbol}: Dhan returned no candle rows.")

    frame = pd.DataFrame(
        {
            "date": epoch_to_ist_date(data["timestamp"]),
            "open": pd.Series(data.get("open", []), dtype="float64"),
            "high": pd.Series(data.get("high", []), dtype="float64"),
            "low": pd.Series(data.get("low", []), dtype="float64"),
            "close": pd.Series(data.get("close", []), dtype="float64"),
            "volume": pd.Series(data.get("volume", []), dtype="float64"),
        }
    )
    frame.insert(0, "symbol", instrument.symbol)
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="last")
    frame["return_pct"] = frame["close"].pct_change(fill_method=None)
    return frame[["symbol", "date", "open", "high", "low", "close", "volume", "return_pct"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch recent Dhan daily candles for the repo ETF universe.")
    parser.add_argument("--access-token", default=os.environ.get("DHAN_ACCESS_TOKEN", ""))
    parser.add_argument("--client-id", default=os.environ.get("DHAN_CLIENT_ID", ""))
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--output-prefix",
        default=str(CACHE_DIR / "dhan_recent"),
        help="Prefix for output files; .csv and _close.csv are appended.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    access_token = str(args.access_token or "").strip()
    if not access_token:
        raise SystemExit("Missing Dhan access token. Pass --access-token or set DHAN_ACCESS_TOKEN.")
    client_id = str(args.client_id or "").strip() or decode_client_id(access_token)
    if not client_id:
        raise SystemExit("Missing Dhan client ID and could not decode it from the token.")

    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    context = DhanContext(client_id, access_token)
    client = dhanhq(context)

    rows: List[pd.DataFrame] = []
    errors: Dict[str, str] = {}
    for instrument in DEFAULT_UNIVERSE.values():
        try:
            rows.append(fetch_symbol_history(client, instrument, args.start_date, args.end_date))
            time.sleep(0.35)
        except Exception as exc:
            errors[instrument.symbol] = f"{type(exc).__name__}: {exc}"

    if errors:
        raise SystemExit(f"Dhan fetch failed for {len(errors)} symbols: {errors}")

    long_frame = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    close_frame = (
        long_frame.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .reindex(columns=list(DEFAULT_UNIVERSE))
    )

    long_path = output_prefix.with_suffix(".csv")
    close_path = output_prefix.parent / f"{output_prefix.name}_close.csv"
    long_frame.to_csv(long_path, index=False)
    close_frame.to_csv(close_path)

    summary = {
        "client_id": client_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": list(DEFAULT_UNIVERSE),
        "rows": int(len(long_frame)),
        "sessions": int(len(close_frame)),
        "long_path": str(long_path),
        "close_path": str(close_path),
        "latest_date": close_frame.index.max().strftime("%Y-%m-%d") if len(close_frame) else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
