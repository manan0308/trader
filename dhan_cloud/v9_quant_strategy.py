#!/usr/bin/env python3
import argparse
import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from dhanhq import dhanhq
from dhanhq.dhan_context import DhanContext


RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]
WARMUP_DAYS = 252


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    execution_frequency: str
    core_weight: float
    tilt_weight: float
    top_n: int
    trade_band: float
    trade_step: float
    crash_floor: float


BASE_V9 = StrategyConfig(
    name="weekly_core85_tilt15",
    execution_frequency="WEEKLY",
    core_weight=0.85,
    tilt_weight=0.15,
    top_n=2,
    trade_band=0.08,
    trade_step=0.75,
    crash_floor=0.70,
)


@dataclass(frozen=True)
class DhanInstrument:
    symbol: str
    security_id: str
    exchange_segment: str = dhanhq.NSE
    instrument: str = "EQUITY"


UNIVERSE: Dict[str, DhanInstrument] = {
    "NIFTYBEES": DhanInstrument("NIFTYBEES", "10576"),
    "MID150BEES": DhanInstrument("MID150BEES", "8506"),
    "HDFCSML250": DhanInstrument("HDFCSML250", "14233"),
    "GOLDBEES": DhanInstrument("GOLDBEES", "14428"),
    "SILVERBEES": DhanInstrument("SILVERBEES", "8080"),
    "MON100": DhanInstrument("MON100", "22739"),
    "LIQUIDBEES": DhanInstrument("LIQUIDBEES", "11006"),
}
ASSET_TO_SYMBOL = {
    "NIFTY": "NIFTYBEES",
    "MIDCAP": "MID150BEES",
    "SMALLCAP": "HDFCSML250",
    "GOLD": "GOLDBEES",
    "SILVER": "SILVERBEES",
    "US": "MON100",
    "CASH": "LIQUIDBEES",
}
SYMBOL_TO_ASSET = {symbol: asset for asset, symbol in ASSET_TO_SYMBOL.items()}
SECURITY_ID_TO_SYMBOL = {instrument.security_id: symbol for symbol, instrument in UNIVERSE.items()}


def decode_client_id(access_token: str) -> Optional[str]:
    try:
        parts = access_token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
        parsed = json.loads(decoded)
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


def fetch_symbol_history(client: dhanhq, instrument: DhanInstrument, start_date: str, end_date: str) -> pd.DataFrame:
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
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
    return frame


def fetch_universe_history(client: dhanhq, lookback_calendar_days: int, end_date: Optional[str]) -> tuple[pd.DataFrame, dict]:
    end_dt = pd.Timestamp(end_date).date() if end_date else date.today()
    start_dt = end_dt - timedelta(days=lookback_calendar_days)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    rows = []
    for instrument in UNIVERSE.values():
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
    metadata = {
        "requested_start_date": start_str,
        "requested_end_date": end_str,
        "latest_market_date": prices.index[-1].strftime("%Y-%m-%d") if len(prices) else None,
        "rows": int(len(prices)),
        "data_source": "dhan_historical_daily",
    }
    return prices, metadata


def schedule_flags(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    stamps = index.to_series()
    if frequency == "WEEKLY":
        bucket = stamps.dt.strftime("%Y-%U")
    elif frequency == "MONTHLY":
        bucket = stamps.dt.to_period("M").astype(str)
    else:
        raise ValueError(f"unsupported execution frequency: {frequency}")
    return bucket.ne(bucket.shift(-1)).fillna(True)


def inverse_vol_weights(vol_row: pd.Series, assets: Iterable[str]) -> pd.Series:
    selected = [asset for asset in assets]
    inv_vol = 1.0 / vol_row[selected].replace(0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv_vol.sum() <= 0:
        inv_vol[:] = 1.0
    return inv_vol / inv_vol.sum()


def build_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    risky_eq = returns[RISKY].mean(axis=1)
    risky_nav = (1.0 + risky_eq).cumprod()
    delta = prices[RISKY].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0.0, np.nan)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    asset_vol20 = returns[RISKY].rolling(20).std() * np.sqrt(252.0)
    asset_vol252 = returns[RISKY].rolling(252).std() * np.sqrt(252.0)
    return {
        "sma50": prices[RISKY].rolling(50).mean(),
        "sma200": prices[RISKY].rolling(200).mean(),
        "mom63": prices[RISKY].pct_change(63, fill_method=None),
        "mom126": prices[RISKY].pct_change(126, fill_method=None),
        "mom252": prices[RISKY].pct_change(252, fill_method=None),
        "vol63": returns[RISKY].rolling(63).std() * np.sqrt(252.0),
        "rsi14": rsi14,
        "asset_dd20": prices[RISKY] / prices[RISKY].rolling(20).max() - 1.0,
        "asset_vol_ratio": asset_vol20 / asset_vol252.replace(0.0, np.nan),
        "breakout252": prices[RISKY] >= prices[RISKY].rolling(252).max().shift(1),
        "cash126": prices["CASH"].pct_change(126, fill_method=None),
        "dd20": risky_nav / risky_nav.rolling(20).max() - 1.0,
        "dd63": risky_nav / risky_nav.rolling(63).max() - 1.0,
        "vol20": risky_eq.rolling(20).std() * np.sqrt(252.0),
        "vol252": risky_eq.rolling(252).std() * np.sqrt(252.0),
        "vol252_median": (risky_eq.rolling(252).std() * np.sqrt(252.0)).expanding().median(),
    }


def crash_signal(features: Dict[str, pd.DataFrame | pd.Series], dt: pd.Timestamp) -> bool:
    dd20 = float(features["dd20"].loc[dt])
    dd63 = float(features["dd63"].loc[dt])
    vol20 = float(features["vol20"].loc[dt])
    vol252 = float(features["vol252"].loc[dt])
    vol252_median = float(features["vol252_median"].loc[dt])
    return bool((dd20 < -0.12 and vol20 > vol252 * 1.35) or (dd63 < -0.10 and vol20 > vol252_median * 1.25))


def breadth_risk_scale(breadth: int) -> float:
    mapping = {0: 0.70, 1: 0.80, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.00, 6: 1.00}
    return mapping.get(breadth, 1.00)


def run_v9_strategy(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    features = build_features(prices)
    schedule = schedule_flags(prices.index, config.execution_frequency)
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = features["sma50"]
    sma200 = features["sma200"]
    mom63 = features["mom63"]
    mom126 = features["mom126"]
    mom252 = features["mom252"]
    vol63 = features["vol63"]
    rsi14 = features["rsi14"]
    asset_dd20 = features["asset_dd20"]
    asset_vol_ratio = features["asset_vol_ratio"]
    breakout252 = features["breakout252"]
    cash126 = features["cash126"]

    for i, dt in enumerate(prices.index):
        if i < WARMUP_DAYS:
            weights.iloc[i] = current
            continue

        trend = prices.loc[dt, RISKY] > sma200.loc[dt]
        strong = trend & (prices.loc[dt, RISKY] > sma50.loc[dt]) & (mom126.loc[dt] > cash126.loc[dt])
        breadth = int(strong.sum())
        crash = crash_signal(features, dt)

        if bool(schedule.loc[dt]):
            risk_scale = breadth_risk_scale(breadth)
            if crash:
                risk_scale = min(risk_scale, config.crash_floor)

            score = (
                0.20 * (mom63.loc[dt] / vol63.loc[dt].replace(0.0, np.nan))
                + 0.35 * mom126.loc[dt]
                + 0.45 * mom252.loc[dt]
            )
            score = score.replace([np.inf, -np.inf], np.nan).fillna(-999.0).sort_values(ascending=False)

            rebound_setup = trend & (asset_dd20.loc[dt] <= -0.08) & (rsi14.loc[dt] <= 45.0)
            score = score + rebound_setup.astype(float) * 0.04
            if bool(trend.get("GOLD", False)) and float(asset_vol_ratio.loc[dt].get("GOLD", 0.0) or 0.0) > 1.35:
                score.loc["GOLD"] = float(score.get("GOLD", -999.0)) + 0.03
            if bool(trend.get("SILVER", False)) and bool(breakout252.loc[dt].get("SILVER", False)):
                score.loc["SILVER"] = float(score.get("SILVER", -999.0)) + 0.03

            selected = [asset for asset in score.index if bool(trend.get(asset, False))][: config.top_n]
            target = pd.Series(0.0, index=ALL)

            core_w = inverse_vol_weights(vol63.loc[dt], RISKY)
            for asset, weight in core_w.items():
                target[asset] += risk_scale * config.core_weight * weight

            if selected and config.tilt_weight > 0:
                tilt_scores = score.loc[selected].clip(lower=0.0)
                if tilt_scores.sum() <= 0:
                    tilt_scores[:] = 1.0
                tilt_scores = tilt_scores / tilt_scores.sum()
                for asset, weight in tilt_scores.items():
                    target[asset] += risk_scale * config.tilt_weight * weight

            if crash and bool(trend.get("GOLD", False)):
                target["GOLD"] += min(0.10, 1.0 - target[RISKY].sum())

            target["CASH"] = max(0.0, 1.0 - target[RISKY].sum())
            target = target / target.sum()

            proposal = current * (1.0 - config.trade_step) + target * config.trade_step
            proposal["CASH"] = max(0.0, 1.0 - proposal[RISKY].sum())
            proposal = proposal / proposal.sum()
            if float((proposal - current).abs().max()) > config.trade_band:
                current = proposal

        if crash and current[RISKY].sum() > 0.80:
            current = current.copy()
            current[RISKY] *= 0.90
            current["CASH"] = 1.0 - current[RISKY].sum()

        weights.iloc[i] = current
    return weights


def first_numeric(mapping: dict, keys: Iterable[str]) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def normalize_holdings(holdings_response: dict) -> Dict[str, int]:
    if str(holdings_response.get("status")) != "success":
        return {}
    rows = holdings_response.get("data") or []
    holdings: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        security_id = str(row.get("securityId") or row.get("securityID") or row.get("security_id") or "")
        symbol = SECURITY_ID_TO_SYMBOL.get(security_id)
        if symbol is None:
            raw_symbol = str(row.get("tradingSymbol") or row.get("customSymbol") or row.get("symbol") or row.get("displayName") or "").upper()
            symbol = raw_symbol if raw_symbol in UNIVERSE else None
        if symbol is None:
            continue
        qty = int(first_numeric(row, ["availableQty", "quantity", "totalQty", "dpQty", "balanceQty", "remainingQuantity", "qty"]))
        if qty > 0:
            holdings[symbol] = qty
    return holdings


def available_cash_inr(funds_response: dict) -> float:
    if str(funds_response.get("status")) != "success":
        return 0.0
    data = funds_response.get("data") or {}
    if not isinstance(data, dict):
        return 0.0
    return max(
        first_numeric(data, ["availabelBalance", "availableBalance"]),
        first_numeric(data, ["withdrawableBalance"]),
        0.0,
    )


def build_rebalance_plan(
    target_weights: pd.Series,
    holdings_qty: Dict[str, int],
    available_cash: float,
    latest_prices: Dict[str, float],
    cash_buffer_pct: float,
    min_order_value: float,
) -> dict:
    holdings_value = sum(float(holdings_qty.get(symbol, 0)) * latest_prices[symbol] for symbol in UNIVERSE if symbol in latest_prices)
    nav = holdings_value + float(available_cash)
    reserve_cash = nav * cash_buffer_pct
    investable_nav = max(nav - reserve_cash, 0.0)

    target_qty: Dict[str, int] = {}
    for asset, weight in target_weights.items():
        symbol = ASSET_TO_SYMBOL[asset]
        price = latest_prices[symbol]
        target_value = investable_nav * max(float(weight), 0.0)
        target_qty[symbol] = int(target_value // price) if price > 0 else 0

    orders = []
    for symbol, instrument in UNIVERSE.items():
        current_qty = int(holdings_qty.get(symbol, 0))
        desired_qty = int(target_qty.get(symbol, 0))
        diff = desired_qty - current_qty
        if diff == 0:
            continue
        side = "BUY" if diff > 0 else "SELL"
        qty = abs(diff)
        est_value = qty * latest_prices[symbol]
        if est_value < min_order_value:
            continue
        orders.append(
            {
                "symbol": symbol,
                "asset": SYMBOL_TO_ASSET[symbol],
                "security_id": instrument.security_id,
                "side": side,
                "quantity": qty,
                "estimated_value_inr": round(est_value, 2),
                "reference_price": round(latest_prices[symbol], 4),
                "current_qty": current_qty,
                "target_qty": desired_qty,
            }
        )

    sells = [order for order in orders if order["side"] == "SELL"]
    buys = [order for order in orders if order["side"] == "BUY"]
    return {
        "available_cash_inr": round(float(available_cash), 2),
        "holdings_value_inr": round(float(holdings_value), 2),
        "nav_inr": round(float(nav), 2),
        "reserve_cash_inr": round(float(reserve_cash), 2),
        "investable_nav_inr": round(float(investable_nav), 2),
        "holdings_qty": holdings_qty,
        "target_qty": target_qty,
        "orders": sells + buys,
    }


def place_rebalance_orders(
    client: dhanhq,
    orders: list[dict],
    *,
    place_orders: bool,
    after_market_order: bool,
    correlation_prefix: str,
    max_order_count: int,
) -> list[dict]:
    if len(orders) > max_order_count:
        raise RuntimeError(f"Refusing to place {len(orders)} orders; max_order_count={max_order_count}")

    results = []
    for idx, order in enumerate(orders, start=1):
        tag = f"{correlation_prefix}_{idx}_{order['side']}_{order['symbol']}"
        if not place_orders:
            results.append({"mode": "dry_run", "tag": tag, **order})
            continue
        response = client.place_order(
            security_id=order["security_id"],
            exchange_segment=dhanhq.NSE,
            transaction_type=getattr(dhanhq, order["side"]),
            quantity=int(order["quantity"]),
            order_type=dhanhq.MARKET,
            product_type=dhanhq.CNC,
            price=0,
            after_market_order=after_market_order,
            validity=dhanhq.DAY,
            tag=tag[:32],
        )
        results.append({"mode": "live", "tag": tag, "request": order, "response": response})
        time.sleep(0.35)
    return results


def format_weights_pct(weights: pd.Series) -> Dict[str, float]:
    return {asset: round(float(weights.get(asset, 0.0)) * 100.0, 2) for asset in ALL}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Dhan Cloud v9 quant strategy with optional order placement.")
    parser.add_argument("--lookback-days", type=int, default=500)
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD override for local testing")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--place-orders", action="store_true", help="Actually place Dhan orders. Default is dry-run.")
    parser.add_argument("--after-market-order", action="store_true", help="Place AMO market orders instead of regular market orders.")
    parser.add_argument("--cash-buffer-pct", type=float, default=0.01, help="Keep this fraction of NAV as raw cash buffer.")
    parser.add_argument("--min-order-value", type=float, default=2000.0, help="Skip tiny orders below this INR value.")
    parser.add_argument("--max-order-count", type=int, default=12)
    args = parser.parse_args()

    print("Starting Dhan Cloud v9 quant strategy execution...")

    try:
        client = build_client()
        prices, metadata = fetch_universe_history(client, args.lookback_days, args.end_date)
        weights = run_v9_strategy(prices, BASE_V9)
        latest_weights = weights.iloc[-1]
        latest_prices = {ASSET_TO_SYMBOL[asset]: float(prices.iloc[-1][asset]) for asset in ALL}

        holdings_response = client.get_holdings()
        funds_response = client.get_fund_limits()
        holdings_qty = normalize_holdings(holdings_response)
        cash_inr = available_cash_inr(funds_response)
        plan = build_rebalance_plan(
            latest_weights,
            holdings_qty,
            cash_inr,
            latest_prices,
            cash_buffer_pct=args.cash_buffer_pct,
            min_order_value=args.min_order_value,
        )
        order_results = place_rebalance_orders(
            client,
            plan["orders"],
            place_orders=args.place_orders,
            after_market_order=args.after_market_order,
            correlation_prefix=f"v9_{int(time.time())}",
            max_order_count=args.max_order_count,
        )

        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Standalone v9 quant target and rebalance plan generated from Dhan daily candles",
            "execution_id": f"exec_{int(time.time())}",
            "model_name": "v9_quant",
            "latest_market_date": metadata.get("latest_market_date"),
            "data_source": metadata.get("data_source"),
            "requested_window": {
                "start": metadata.get("requested_start_date"),
                "end": metadata.get("requested_end_date"),
                "rows": metadata.get("rows"),
            },
            "target_weights_pct": format_weights_pct(latest_weights),
            "reference_prices_inr": {symbol: round(price, 4) for symbol, price in latest_prices.items()},
            "rebalance_plan": plan,
            "order_mode": "live" if args.place_orders else "dry_run",
            "after_market_order": bool(args.after_market_order),
            "order_results": order_results,
            "broker_caveats": [
                "Dhan holdings sells may require eDIS/TPIN authorization depending on broker-side state.",
                "Sizing uses latest Dhan daily close as the reference price, not live LTP.",
                "The strategy maps CASH to LIQUIDBEES; a small raw cash buffer is kept separately.",
            ],
        }

        print(f"Result: {json.dumps(payload, indent=2)}")
        if args.output_json:
            path = Path(args.output_json).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    except Exception as e:
        print(f"Error in strategy execution: {str(e)}")
        raise

    print("Dhan Cloud v9 quant strategy completed successfully")


if __name__ == "__main__":
    main()
