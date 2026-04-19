#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from dhanhq import dhanhq
from dhanhq.dhan_context import DhanContext


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy.v9_engine import ALL  # noqa: E402


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

SYMBOL_TO_ASSET = {
    "NIFTYBEES": "NIFTY",
    "MID150BEES": "MIDCAP",
    "HDFCSML250": "SMALLCAP",
    "GOLDBEES": "GOLD",
    "SILVERBEES": "SILVER",
    "MON100": "US",
    "LIQUIDBEES": "CASH",
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


def get_env_access_token() -> str:
    for key in ("DHAN_ACCESS_TOKEN", "DHAN_TOKEN", "DHAN_API_TOKEN"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    raise RuntimeError("Missing Dhan access token. Set DHAN_ACCESS_TOKEN or DHAN_TOKEN.")


def get_env_client_id(access_token: str) -> str:
    for key in ("DHAN_CLIENT_ID", "DHAN_CLIENTID"):
        value = str(os.environ.get(key, "")).strip()
        if value:
            return value
    decoded = decode_client_id(access_token)
    if decoded:
        return decoded
    raise RuntimeError("Missing Dhan client id. Set DHAN_CLIENT_ID.")


def build_client() -> dhanhq:
    access_token = get_env_access_token()
    client_id = get_env_client_id(access_token)
    return dhanhq(DhanContext(client_id, access_token))


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
        raise RuntimeError(f"{instrument.symbol}: no candle rows returned")

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
    return frame


def fetch_universe_history(
    *,
    lookback_calendar_days: int = 500,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    client = build_client()

    end_dt = pd.Timestamp(end_date).date() if end_date else date.today()
    start_dt = end_dt - timedelta(days=lookback_calendar_days)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    rows = []
    for instrument in DEFAULT_UNIVERSE.values():
        rows.append(fetch_symbol_history(client, instrument, start_str, end_str))
        time.sleep(0.35)

    long_frame = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    long_frame["asset"] = long_frame["symbol"].map(SYMBOL_TO_ASSET)

    prices = (
        long_frame.pivot(index="date", columns="asset", values="close")
        .sort_index()
        .reindex(columns=ALL)
        .dropna()
    )
    volume = (
        long_frame.pivot(index="date", columns="asset", values="volume")
        .sort_index()
        .reindex(index=prices.index, columns=ALL)
        .fillna(0.0)
    )
    metadata = {
        "requested_start_date": start_str,
        "requested_end_date": end_str,
        "latest_market_date": prices.index[-1].strftime("%Y-%m-%d") if len(prices) else None,
        "rows": int(len(prices)),
        "symbols": list(DEFAULT_UNIVERSE),
        "data_source": "dhan_historical_daily",
    }
    return prices, volume, metadata


def format_weights_pct(weights: pd.Series) -> Dict[str, float]:
    return {asset: round(float(weights.get(asset, 0.0)) * 100.0, 2) for asset in ALL}


def write_payload_if_requested(payload: dict, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
