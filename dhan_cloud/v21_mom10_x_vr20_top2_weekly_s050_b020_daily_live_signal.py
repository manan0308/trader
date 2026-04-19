#!/usr/bin/env python3
import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
from urllib import error, request

import numpy as np
import pandas as pd


MODEL_NAME = "v21_mom10_x_vr20_top2_weekly_s050_b020_daily_live_signal"
RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]
WARMUP_DAYS = 252
LOOKBACK_CALENDAR_DAYS = 500
END_DATE_OVERRIDE = "{{END_DATE_OVERRIDE}}"
ACCESS_TOKEN = "{{DHAN_ACCESS_TOKEN}}"
DHAN_HISTORICAL_URL = "https://api.dhan.co/v2/charts/historical"

UNIVERSE = {
    "NIFTY": {"symbol": "NIFTYBEES", "securityId": "10576", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
    "MIDCAP": {"symbol": "MID150BEES", "securityId": "8506", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
    "SMALLCAP": {"symbol": "HDFCSML250", "securityId": "14233", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
    "GOLD": {"symbol": "GOLDBEES", "securityId": "14428", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
    "SILVER": {"symbol": "SILVERBEES", "securityId": "8080", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
    "US": {"symbol": "MON100", "securityId": "22739", "exchangeSegment": "NSE_EQ", "instrument": "EQUITY"},
}


def unresolved_template(value: str) -> bool:
    text = str(value).strip()
    return text.startswith("{{") and text.endswith("}}")


def resolve_int(default_value: int, raw_value: str) -> int:
    if unresolved_template(raw_value):
        return int(default_value)
    try:
        return int(float(raw_value))
    except Exception:
        return int(default_value)


def resolve_end_date() -> date:
    if unresolved_template(END_DATE_OVERRIDE) or str(END_DATE_OVERRIDE).strip() == "":
        return date.today()
    return pd.Timestamp(END_DATE_OVERRIDE).date()


def resolve_access_token() -> str:
    if unresolved_template(ACCESS_TOKEN) or str(ACCESS_TOKEN).strip() == "":
        raise RuntimeError("Missing DHAN_ACCESS_TOKEN strategy variable.")
    return str(ACCESS_TOKEN).strip()


def epoch_to_date(epoch_values: List[float]) -> pd.Series:
    return (
        pd.to_datetime(pd.Series(epoch_values, dtype="float64"), unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
        .dt.normalize()
    )


def fetch_daily_history_for_asset(asset: str, access_token: str, start_date: str, end_date: str) -> pd.DataFrame:
    meta = UNIVERSE[asset]
    payload = {
        "securityId": meta["securityId"],
        "exchangeSegment": meta["exchangeSegment"],
        "instrument": meta["instrument"],
        "expiryCode": 0,
        "oi": False,
        "fromDate": start_date,
        "toDate": end_date,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        DHAN_HISTORICAL_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": access_token,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError("Dhan historical fetch failed for " + asset + ": " + detail) from exc
    except Exception as exc:
        raise RuntimeError("Dhan historical fetch failed for " + asset + ": " + str(exc)) from exc

    parsed = json.loads(raw)
    timestamps = parsed.get("timestamp", [])
    if not timestamps:
        raise RuntimeError("No historical rows returned for " + asset)

    frame = pd.DataFrame(
        {
            "date": epoch_to_date(timestamps),
            "asset": asset,
            "close": pd.Series(parsed.get("close", []), dtype="float64"),
            "volume": pd.Series(parsed.get("volume", []), dtype="float64"),
        }
    )
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["date", "asset"], keep="last")


def fetch_dhan_history() -> Tuple[pd.DataFrame, pd.DataFrame]:
    access_token = resolve_access_token()
    end_dt = resolve_end_date()
    start_dt = end_dt - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    rows = []
    for asset in RISKY:
        rows.append(fetch_daily_history_for_asset(asset, access_token, start_str, end_str))

    frame = pd.concat(rows, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    prices = (
        frame.pivot(index="date", columns="asset", values="close")
        .sort_index()
        .reindex(columns=RISKY)
        .dropna()
    )
    volume = (
        frame.pivot(index="date", columns="asset", values="volume")
        .sort_index()
        .reindex(index=prices.index, columns=RISKY)
        .fillna(0.0)
    )
    if prices.empty:
        raise RuntimeError("No valid price matrix could be built from Dhan historical API.")
    return prices, volume


def schedule_flags(index: pd.DatetimeIndex) -> pd.Series:
    stamps = index.to_series()
    bucket = stamps.dt.strftime("%Y-%U")
    return bucket.ne(bucket.shift(-1)).fillna(True)


def build_mom10_x_vr20_scores(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    mom10 = prices.pct_change(10, fill_method=None)
    volume_ratio20 = volume / volume.rolling(20).median().replace(0.0, np.nan)
    return mom10 * volume_ratio20


def build_top_n_targets(score_frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    targets = pd.DataFrame(0.0, index=score_frame.index, columns=ALL)
    for i, dt in enumerate(score_frame.index):
        if i < WARMUP_DAYS:
            targets.loc[dt, "CASH"] = 1.0
            continue
        score = score_frame.loc[dt].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        picks = list(score.index[:top_n])
        if not picks:
            targets.loc[dt, "CASH"] = 1.0
            continue
        weight = 1.0 / float(len(picks))
        for asset in picks:
            targets.loc[dt, asset] = weight
    return targets


def apply_rebalance_policy(targets: pd.DataFrame, trade_band: float, trade_step: float) -> pd.DataFrame:
    schedule = schedule_flags(targets.index)
    weights = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    current = pd.Series(0.0, index=targets.columns)
    current["CASH"] = 1.0

    for i, dt in enumerate(targets.index):
        target = targets.loc[dt].copy()
        if i < WARMUP_DAYS:
            weights.loc[dt] = current
            continue

        if bool(schedule.loc[dt]):
            proposal = current * (1.0 - trade_step) + target * trade_step
            proposal = proposal.clip(lower=0.0)
            if float(proposal.sum()) > 0.0:
                proposal = proposal / float(proposal.sum())
            else:
                proposal = target
            if float((proposal - current).abs().max()) > trade_band:
                current = proposal
        weights.loc[dt] = current
    return weights


def format_weights_pct(mapping: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for asset in ALL:
        out[asset] = round(float(mapping.get(asset, 0.0)) * 100.0, 2)
    return out


def top_scores_as_dict(score_row: pd.Series, top_n: int) -> Dict[str, float]:
    cleaned = score_row.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    return {asset: round(float(value), 6) for asset, value in cleaned.head(top_n).items()}


def main() -> None:
    print("Starting Dhan live daily signal computation...")

    prices, volume = fetch_dhan_history()
    if len(prices) <= WARMUP_DAYS:
        raise RuntimeError("Need more history rows. This model requires more than 252 trading rows.")

    scores = build_mom10_x_vr20_scores(prices, volume)
    raw_targets = build_top_n_targets(scores, top_n=2)
    smoothed_weights = apply_rebalance_policy(raw_targets, 0.20, 0.50)

    latest_market_date = prices.index[-1].strftime("%Y-%m-%d")
    latest_raw_target = raw_targets.iloc[-1].to_dict()
    latest_live_target = smoothed_weights.iloc[-1].to_dict()
    latest_scores = scores.iloc[-1]
    previous_live_target = smoothed_weights.iloc[-2].to_dict() if len(smoothed_weights) >= 2 else latest_live_target
    changed_assets = {}
    for asset in ALL:
        delta = float(latest_live_target.get(asset, 0.0)) - float(previous_live_target.get(asset, 0.0))
        if abs(delta) > 1e-10:
            changed_assets[asset] = round(delta * 100.0, 2)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "message": "Daily live v21 signal generated from Dhan historical API",
        "model_name": MODEL_NAME,
        "latest_market_date": latest_market_date,
        "input_dates": int(len(prices.index)),
        "reference_prices": {asset: round(float(prices.iloc[-1][asset]), 4) for asset in RISKY},
        "raw_top2_target_weights_pct": format_weights_pct(latest_raw_target),
        "smoothed_live_target_weights_pct": format_weights_pct(latest_live_target),
        "previous_smoothed_target_weights_pct": format_weights_pct(previous_live_target),
        "changed_target_weights_pct": changed_assets,
        "rebalance_changed": bool(changed_assets),
        "top_score_assets": top_scores_as_dict(latest_scores, 4),
        "notes": [
            "This is a live daily signal runner, not a historical backtest report.",
            "It pulls recent daily candles from Dhan each run and computes the latest weekly-smoothed v21 target.",
            "No live orders are placed in this version.",
        ],
    }

    print(json.dumps(payload, indent=2))
    print("Dhan live daily signal computation completed successfully")


if __name__ == "__main__":
    main()
