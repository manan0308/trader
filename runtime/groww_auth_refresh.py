#!/usr/bin/env python3
"""
Groww auth bootstrap / refresh
==============================

Purpose:
- mint a fresh Groww access token on the VPS each morning
- verify that the token actually works
- write the fresh token into config/runtime.env for the rest of the stack
- publish a non-secret auth-status artifact for the dashboard and APIs
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import tempfile
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Dict
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from broker.groww_client import (
    GROWW_ACCESS_TOKEN_ENV,
    GROWW_API_KEY_ENV,
    GROWW_API_SECRET_ENV,
    GROWW_TOTP_CODE_ENV,
    GROWW_TOTP_SECRET_ENV,
    GrowwAPI,
    GrowwSession,
    generate_totp_code,
    load_production_groww_universe,
)
from strategy.v9_engine import CACHE_DIR
from runtime.env_loader import BASE_DIR, load_runtime_env
from runtime.store import GROWW_AUTH_STATUS_PATH, write_json


IST = ZoneInfo("Asia/Kolkata")
RUNTIME_ENV_PATH = BASE_DIR / "config" / "runtime.env"
MANAGED_ENV_KEYS = {
    GROWW_ACCESS_TOKEN_ENV,
    "GROWW_ACCESS_TOKEN_REFRESHED_AT",
    "GROWW_ACCESS_TOKEN_SOURCE",
    "GROWW_ACCESS_TOKEN_ASSUMED_EXPIRY_IST",
    "GROWW_LAST_PUBLIC_IP",
}


def now_ist() -> datetime:
    return datetime.now(tz=IST)


def assumed_expiry_ist(ref: datetime) -> datetime:
    today_six = datetime.combine(ref.date(), dt_time(hour=6, minute=0), tzinfo=IST)
    if ref < today_six:
        return today_six
    return today_six + timedelta(days=1)


def recommended_refresh_ist(ref: datetime) -> datetime:
    target = datetime.combine(ref.date(), dt_time(hour=8, minute=35), tzinfo=IST)
    if ref < target:
        return target
    return target + timedelta(days=1)


def discover_public_ipv4() -> str | None:
    candidates = [
        "https://ipv4.icanhazip.com",
        "https://api.ipify.org?format=text",
        "https://ifconfig.me/ip",
    ]
    for url in candidates:
        try:
            with urlopen(url, timeout=5) as response:  # nosec - operational helper
                value = response.read().decode("utf-8").strip()
                if value and ipaddress.ip_address(value).version == 4:
                    return value
        except Exception:
            continue
    return None


def normalize_access_token(payload: object) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        return str(
            payload.get("access_token")
            or payload.get("token")
            or payload.get("jwtToken")
            or ""
        ).strip()
    return ""


def render_runtime_env(existing: Dict[str, str], updates: Dict[str, str]) -> str:
    merged = {k: v for k, v in existing.items() if k not in MANAGED_ENV_KEYS}
    merged.update({k: v for k, v in updates.items() if v})
    lines = [f"{key}={json.dumps(value)}" for key, value in sorted(merged.items())]
    return "\n".join(lines) + ("\n" if lines else "")


def read_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def persist_runtime_token(path: Path, token: str, *, source: str, refreshed_at: datetime, public_ip: str | None) -> None:
    existing = read_env_file(path)
    updates = {
        GROWW_ACCESS_TOKEN_ENV: token,
        "GROWW_ACCESS_TOKEN_REFRESHED_AT": refreshed_at.isoformat(),
        "GROWW_ACCESS_TOKEN_SOURCE": source,
        "GROWW_ACCESS_TOKEN_ASSUMED_EXPIRY_IST": assumed_expiry_ist(refreshed_at).isoformat(),
        "GROWW_LAST_PUBLIC_IP": public_ip or "",
    }
    atomic_write_text(path, render_runtime_env(existing, updates))


def choose_auth_mode() -> tuple[str, str | None]:
    load_runtime_env(override=True)
    api_key = os.getenv(GROWW_API_KEY_ENV)
    api_secret = os.getenv(GROWW_API_SECRET_ENV)
    totp_secret = os.getenv(GROWW_TOTP_SECRET_ENV)
    totp_code = os.getenv(GROWW_TOTP_CODE_ENV)
    access_token = os.getenv(GROWW_ACCESS_TOKEN_ENV)

    if api_key and (totp_secret or totp_code):
        return "api_key_totp", api_key
    if api_key and api_secret:
        return "api_key_secret", api_key
    if access_token:
        return "access_token_only", None
    return "missing", None


def mint_access_token() -> tuple[str, str]:
    mode, api_key = choose_auth_mode()
    if mode == "api_key_totp":
        code = os.getenv(GROWW_TOTP_CODE_ENV) or generate_totp_code(os.getenv(GROWW_TOTP_SECRET_ENV, ""))
        token = normalize_access_token(GrowwAPI.get_access_token(api_key=api_key or "", totp=code))
        if not token:
            raise RuntimeError("Groww TOTP flow returned an empty access token.")
        return token, mode
    if mode == "api_key_secret":
        token = normalize_access_token(
            GrowwAPI.get_access_token(
                api_key=api_key or "",
                secret=os.getenv(GROWW_API_SECRET_ENV),
            )
        )
        if not token:
            raise RuntimeError("Groww secret flow returned an empty access token.")
        return token, mode
    if mode == "access_token_only":
        token = os.getenv(GROWW_ACCESS_TOKEN_ENV, "").strip()
        if not token:
            raise RuntimeError("GROWW_ACCESS_TOKEN is empty.")
        return token, mode
    raise RuntimeError(
        "Missing Groww credentials. Set GROWW_API_KEY plus GROWW_TOTP_SECRET or GROWW_API_SECRET, "
        "or provide GROWW_ACCESS_TOKEN manually."
    )


def smoke_test_token(token: str) -> dict:
    session = GrowwSession(
        access_token=token,
        instrument_map=load_production_groww_universe(),
        cache_dir=CACHE_DIR,
    )
    profile = session.client.get_user_profile()
    historical_candles_ok = False
    historical_candles_error = None
    start = (now_ist() - timedelta(days=14)).strftime("%Y-%m-%d 09:15:00")
    end = now_ist().strftime("%Y-%m-%d 15:30:00")

    try:
        candles = session.client.get_historical_candles(
            exchange=session.client.EXCHANGE_NSE,
            segment=session.client.SEGMENT_CASH,
            groww_symbol=session.instrument_map["NIFTY"].groww_symbol,
            start_time=start,
            end_time=end,
            candle_interval=session.client.CANDLE_INTERVAL_DAY,
        )
        historical_candles_ok = bool((candles or {}).get("candles"))
        if not historical_candles_ok:
            historical_candles_error = "Groww returned no candles for the historical smoke test."
    except Exception as exc:
        historical_candles_error = f"{type(exc).__name__}: {exc}"

    return {
        "ucc": profile.get("ucc"),
        "vendor_user_id": profile.get("vendor_user_id"),
        "segments": profile.get("active_segments", []),
        "nse_enabled": profile.get("nse_enabled"),
        "bse_enabled": profile.get("bse_enabled"),
        "ddpi_enabled": profile.get("ddpi_enabled"),
        "historical_candles_ok": historical_candles_ok,
        "historical_candles_error": historical_candles_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint or verify Groww auth, persist the active token, and publish auth status.")
    parser.add_argument("--status-output", default=str(GROWW_AUTH_STATUS_PATH))
    parser.add_argument("--runtime-env", default=str(RUNTIME_ENV_PATH))
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--no-write-runtime-env", action="store_true")
    args = parser.parse_args()

    status_path = Path(args.status_output).expanduser().resolve()
    runtime_env_path = Path(args.runtime_env).expanduser().resolve()
    refreshed_at = now_ist()
    public_ip = discover_public_ipv4()

    if GrowwAPI is None:
        raise RuntimeError("growwapi is not installed in this environment.")

    try:
        token, source = mint_access_token()
        smoke = {} if args.skip_smoke_test else smoke_test_token(token)
        if not args.no_write_runtime_env:
            persist_runtime_token(
                runtime_env_path,
                token,
                source=source,
                refreshed_at=refreshed_at,
                public_ip=public_ip,
            )
        status = {
            "status": "ok",
            "updated_at": refreshed_at.isoformat(),
            "source": source,
            "token_present": bool(token),
            "token_refreshed": source != "access_token_only",
            "runtime_env_written": not args.no_write_runtime_env,
            "public_ipv4": public_ip,
            "assumed_token_expiry_ist": assumed_expiry_ist(refreshed_at).isoformat(),
            "recommended_refresh_ist": recommended_refresh_ist(refreshed_at).isoformat(),
            "smoke_test_ok": True,
            "profile": smoke,
            "historical_candles_ok": bool(smoke.get("historical_candles_ok", False)),
            "historical_candles_error": smoke.get("historical_candles_error"),
            "notes": [
                "Groww auth is broker-controlled; refresh each morning before market open.",
                "If TOTP is configured, this flow can mint a fresh token programmatically.",
            ],
        }
        write_json(status_path, status)
        print(json.dumps(status, indent=2))
    except Exception as exc:
        failure = {
            "status": "error",
            "updated_at": refreshed_at.isoformat(),
            "public_ipv4": public_ip,
            "source": choose_auth_mode()[0],
            "smoke_test_ok": False,
            "runtime_env_written": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_json(status_path, failure)
        print(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
