#!/usr/bin/env python3
import base64
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dhanhq import dhanhq
from dhanhq.dhan_context import DhanContext


RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]
WARMUP_DAYS = 252
MODEL_NAME = "v21_mom10_x_vr20_top2_weekly_s050_b020_live_paper"
ACCESS_TOKEN = ""
CLIENT_ID = ""
LOOKBACK_DAYS = 500
END_DATE_OVERRIDE = None
INITIAL_CASH_INR = 1_000_000.0
PLACE_ORDERS = False
AFTER_MARKET_ORDER = False
CASH_BUFFER_PCT = 0.01
MIN_ORDER_VALUE = 2000.0
MAX_ORDER_COUNT = 12
STATE_JSON_PATH = "/tmp/v21_momvol_weekly_live_paper_state.json"
PAPER_JOURNAL_PATH = "/tmp/v21_momvol_weekly_paper_journal.jsonl"
ORDER_JOURNAL_PATH = "/tmp/v21_momvol_weekly_live_order_journal.jsonl"
OUTPUT_JSON_PATH = "/tmp/v21_momvol_weekly_live_paper_output.json"


@dataclass(frozen=True)
class DhanInstrument:
    symbol: str
    security_id: str
    exchange_segment: str = dhanhq.NSE
    instrument: str = "EQUITY"


@dataclass(frozen=True)
class DeliveryCostBreakdown:
    notional: float
    side: str
    brokerage: float
    stt: float
    stamp_duty: float
    exchange_transaction: float
    sebi_turnover: float
    ipft: float
    dp_charge: float
    gst: float

    @property
    def total(self) -> float:
        return float(
            self.brokerage
            + self.stt
            + self.stamp_duty
            + self.exchange_transaction
            + self.sebi_turnover
            + self.ipft
            + self.dp_charge
            + self.gst
        )


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


def resolve_access_token(cli_value: Optional[str] = None) -> str:
    if cli_value:
        return str(cli_value).strip()
    if str(ACCESS_TOKEN).strip():
        return str(ACCESS_TOKEN).strip()
    raise RuntimeError("Missing Dhan access token. Pass --access-token or set ACCESS_TOKEN at the top of the script.")


def resolve_client_id(access_token: str, cli_value: Optional[str] = None) -> str:
    if cli_value:
        return str(cli_value).strip()
    if str(CLIENT_ID).strip():
        return str(CLIENT_ID).strip()
    decoded = decode_client_id(access_token)
    if decoded:
        return decoded
    raise RuntimeError("Missing Dhan client id. Pass --client-id or set CLIENT_ID at the top of the script.")


def build_client(access_token: Optional[str] = None, client_id: Optional[str] = None) -> dhanhq:
    access_token = resolve_access_token(access_token)
    client_id = resolve_client_id(access_token, client_id)
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
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(["symbol", "date"], keep="last")


def fetch_universe_history(client: dhanhq, lookback_calendar_days: int, end_date: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
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
        "data_source": "dhan_historical_daily",
    }
    return prices, volume, metadata


def schedule_flags(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    stamps = index.to_series()
    if frequency == "WEEKLY":
        bucket = stamps.dt.strftime("%Y-%U")
    elif frequency == "MONTHLY":
        bucket = stamps.dt.to_period("M").astype(str)
    else:
        raise ValueError(f"unsupported execution frequency: {frequency}")
    return bucket.ne(bucket.shift(-1)).fillna(True)


def build_mom10_x_vr20_scores(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    mom10 = prices[RISKY].pct_change(10, fill_method=None)
    volume_ratio20 = volume[RISKY] / volume[RISKY].rolling(20).median().replace(0.0, np.nan)
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
        for asset in picks:
            targets.loc[dt, asset] = 1.0 / len(picks)
    return targets


def apply_rebalance_policy(targets: pd.DataFrame, frequency: str, trade_band: float, trade_step: float) -> pd.DataFrame:
    if frequency == "DAILY":
        schedule = pd.Series(True, index=targets.index)
    elif frequency == "WEEKLY":
        schedule = schedule_flags(targets.index, "WEEKLY")
    else:
        raise ValueError(f"unsupported frequency: {frequency}")

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


def order_cost(notional: float, side: str) -> DeliveryCostBreakdown:
    value = max(float(notional), 0.0)
    normalized_side = str(side).upper()
    if value <= 0.0:
        return DeliveryCostBreakdown(0.0, normalized_side, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    brokerage = min(20.0, value * 0.001)
    stt = value * 0.001
    stamp = value * 0.00015 if normalized_side == "BUY" else 0.0
    exchange = value * 0.0000297
    sebi = value * 0.000001
    ipft = value * 0.000001
    dp_charge = 20.0 if normalized_side == "SELL" and value >= 100.0 else 0.0
    gst = (brokerage + exchange + sebi + ipft + dp_charge) * 0.18
    return DeliveryCostBreakdown(
        notional=value,
        side=normalized_side,
        brokerage=brokerage,
        stt=stt,
        stamp_duty=stamp,
        exchange_transaction=exchange,
        sebi_turnover=sebi,
        ipft=ipft,
        dp_charge=dp_charge,
        gst=gst,
    )


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
        symbol = None
        for known_symbol, instrument in UNIVERSE.items():
            if instrument.security_id == security_id:
                symbol = known_symbol
                break
        if symbol is None:
            raw_symbol = str(
                row.get("tradingSymbol")
                or row.get("customSymbol")
                or row.get("symbol")
                or row.get("displayName")
                or ""
            ).upper()
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


def build_live_rebalance_plan(target_weights: pd.Series, holdings_qty: Dict[str, int], available_cash: float, latest_prices: Dict[str, float], cash_buffer_pct: float, min_order_value: float) -> dict:
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


def place_live_orders(client: dhanhq, orders: List[dict], place_orders: bool, after_market_order: bool, correlation_prefix: str, max_order_count: int) -> List[dict]:
    if len(orders) > max_order_count:
        raise RuntimeError(f"Refusing to place {len(orders)} orders; max_order_count={max_order_count}")

    transaction_type_map = {
        "BUY": dhanhq.BUY,
        "SELL": dhanhq.SELL,
    }
    results = []
    for idx, order in enumerate(orders, start=1):
        tag = f"{correlation_prefix}_{idx}_{order['side']}_{order['symbol']}"
        if not place_orders:
            results.append({"mode": "dry_run", "tag": tag, **order})
            continue
        response = client.place_order(
            security_id=order["security_id"],
            exchange_segment=dhanhq.NSE,
            transaction_type=transaction_type_map[order["side"]],
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


def read_json_file(path: str) -> dict:
    validate_tmp_path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str, payload: dict) -> None:
    validate_tmp_path(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl_file(path: str, payload: dict) -> None:
    validate_tmp_path(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def validate_tmp_path(path: str) -> None:
    normalized = str(path or "").strip()
    if not normalized.startswith("/tmp/"):
        raise RuntimeError("This runner only allows file writes under /tmp/.")


def default_state(initial_cash: float) -> dict:
    return {
        "model_name": MODEL_NAME,
        "initial_cash_inr": float(initial_cash),
        "cash_inr": float(initial_cash),
        "holdings_qty": {symbol: 0 for symbol in UNIVERSE},
        "last_processed_market_date": None,
        "last_signal_hash": None,
        "last_live_processed_market_date": None,
        "last_live_signal_hash": None,
        "last_live_execution_id": None,
        "last_nav_inr": float(initial_cash),
        "runs": 0,
        "created_at": datetime.now().isoformat(),
    }


def load_state(path: str, initial_cash: float) -> dict:
    try:
        state = read_json_file(path)
    except FileNotFoundError:
        return default_state(initial_cash)
    holdings = {symbol: int((state.get("holdings_qty") or {}).get(symbol, 0)) for symbol in UNIVERSE}
    return {
        "model_name": state.get("model_name", MODEL_NAME),
        "initial_cash_inr": float(state.get("initial_cash_inr", initial_cash)),
        "cash_inr": float(state.get("cash_inr", initial_cash)),
        "holdings_qty": holdings,
        "last_processed_market_date": state.get("last_processed_market_date"),
        "last_signal_hash": state.get("last_signal_hash"),
        "last_live_processed_market_date": state.get("last_live_processed_market_date"),
        "last_live_signal_hash": state.get("last_live_signal_hash"),
        "last_live_execution_id": state.get("last_live_execution_id"),
        "last_nav_inr": float(state.get("last_nav_inr", initial_cash)),
        "runs": int(state.get("runs", 0)),
        "created_at": state.get("created_at") or datetime.now().isoformat(),
    }


def save_state(path: str, state: dict) -> None:
    write_json_file(path, state)


def append_journal(path: str, row: dict) -> None:
    append_jsonl_file(path, row)


def holdings_value_inr(holdings_qty: Dict[str, int], latest_prices: Dict[str, float]) -> float:
    return float(sum(int(holdings_qty.get(symbol, 0)) * float(latest_prices[symbol]) for symbol in UNIVERSE))


def book_weights_pct(cash_inr: float, holdings_qty: Dict[str, int], latest_prices: Dict[str, float]) -> Dict[str, float]:
    asset_values = {asset: 0.0 for asset in ALL}
    for symbol, qty in holdings_qty.items():
        asset = SYMBOL_TO_ASSET[symbol]
        asset_values[asset] += float(qty) * float(latest_prices[symbol])
    total = float(cash_inr) + sum(asset_values.values())
    if total <= 0.0:
        return {asset: 0.0 for asset in ALL}
    return {asset: round(100.0 * asset_values[asset] / total, 2) for asset in ALL}


def target_qty_from_weights(target_weights: pd.Series, nav_inr: float, latest_prices: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for asset in ALL:
        symbol = ASSET_TO_SYMBOL[asset]
        price = float(latest_prices[symbol])
        target_value = float(target_weights.get(asset, 0.0)) * max(nav_inr, 0.0)
        out[symbol] = int(target_value // price) if price > 0 else 0
    return out


def max_affordable_buy_qty(cash_inr: float, price: float, desired_qty: int) -> int:
    low, high = 0, int(max(desired_qty, 0))
    while low < high:
        mid = (low + high + 1) // 2
        notional = mid * price
        total = notional + order_cost(notional, "BUY").total
        if total <= cash_inr + 1e-9:
            low = mid
        else:
            high = mid - 1
    return low


def execute_paper_rebalance(state: dict, target_weights: pd.Series, latest_prices: Dict[str, float]) -> dict:
    holdings_qty = {symbol: int(state["holdings_qty"].get(symbol, 0)) for symbol in UNIVERSE}
    cash_inr = float(state["cash_inr"])
    nav_before = cash_inr + holdings_value_inr(holdings_qty, latest_prices)
    target_qty = target_qty_from_weights(target_weights, nav_before, latest_prices)

    planned_orders = []
    fills = []
    total_cost_inr = 0.0

    for symbol in UNIVERSE:
        current_qty = holdings_qty.get(symbol, 0)
        desired_qty = target_qty.get(symbol, 0)
        diff = desired_qty - current_qty
        if diff == 0:
            continue
        side = "BUY" if diff > 0 else "SELL"
        planned_orders.append(
            {
                "symbol": symbol,
                "asset": SYMBOL_TO_ASSET[symbol],
                "side": side,
                "quantity": int(abs(diff)),
                "reference_price": round(float(latest_prices[symbol]), 4),
            }
        )

    for order in [o for o in planned_orders if o["side"] == "SELL"]:
        qty = min(int(order["quantity"]), int(holdings_qty.get(order["symbol"], 0)))
        if qty <= 0:
            continue
        price = float(latest_prices[order["symbol"]])
        notional = qty * price
        cost = order_cost(notional, "SELL")
        cash_inr += notional - cost.total
        holdings_qty[order["symbol"]] = int(holdings_qty.get(order["symbol"], 0)) - qty
        total_cost_inr += cost.total
        fills.append(
            {
                **order,
                "executed_quantity": qty,
                "notional_inr": round(notional, 2),
                "cost_inr": round(cost.total, 2),
                "cost_breakdown": {
                    "brokerage": round(cost.brokerage, 2),
                    "stt": round(cost.stt, 2),
                    "stamp_duty": round(cost.stamp_duty, 2),
                    "exchange_transaction": round(cost.exchange_transaction, 2),
                    "sebi_turnover": round(cost.sebi_turnover, 2),
                    "ipft": round(cost.ipft, 2),
                    "dp_charge": round(cost.dp_charge, 2),
                    "gst": round(cost.gst, 2),
                },
            }
        )

    for order in [o for o in planned_orders if o["side"] == "BUY"]:
        price = float(latest_prices[order["symbol"]])
        desired_qty = int(order["quantity"])
        qty = max_affordable_buy_qty(cash_inr, price, desired_qty)
        if qty <= 0:
            fills.append({**order, "executed_quantity": 0, "notional_inr": 0.0, "cost_inr": 0.0, "skipped": "insufficient_cash_after_costs"})
            continue
        notional = qty * price
        cost = order_cost(notional, "BUY")
        cash_inr -= notional + cost.total
        holdings_qty[order["symbol"]] = int(holdings_qty.get(order["symbol"], 0)) + qty
        total_cost_inr += cost.total
        fills.append(
            {
                **order,
                "executed_quantity": qty,
                "notional_inr": round(notional, 2),
                "cost_inr": round(cost.total, 2),
                "cost_breakdown": {
                    "brokerage": round(cost.brokerage, 2),
                    "stt": round(cost.stt, 2),
                    "stamp_duty": round(cost.stamp_duty, 2),
                    "exchange_transaction": round(cost.exchange_transaction, 2),
                    "sebi_turnover": round(cost.sebi_turnover, 2),
                    "ipft": round(cost.ipft, 2),
                    "dp_charge": round(cost.dp_charge, 2),
                    "gst": round(cost.gst, 2),
                },
            }
        )

    nav_after = cash_inr + holdings_value_inr(holdings_qty, latest_prices)
    return {
        "cash_inr": round(cash_inr, 2),
        "holdings_qty": holdings_qty,
        "nav_before_inr": round(nav_before, 2),
        "nav_after_inr": round(nav_after, 2),
        "target_qty": target_qty,
        "planned_orders": planned_orders,
        "fills": fills,
        "total_cost_inr": round(total_cost_inr, 2),
    }


def format_weights_pct(weights: Union[pd.Series, Dict[str, float]]) -> Dict[str, float]:
    if isinstance(weights, pd.Series):
        mapping = weights.to_dict()
    else:
        mapping = dict(weights)
    return {asset: round(float(mapping.get(asset, 0.0)) * 100.0, 2) for asset in ALL}


def holdings_snapshot(holdings_qty: Dict[str, int], latest_prices: Dict[str, float]) -> Dict[str, dict]:
    out = {}
    for symbol in UNIVERSE:
        qty = int(holdings_qty.get(symbol, 0))
        price = float(latest_prices[symbol])
        out[symbol] = {
            "asset": SYMBOL_TO_ASSET[symbol],
            "quantity": qty,
            "price_inr": round(price, 4),
            "market_value_inr": round(qty * price, 2),
        }
    return out


def signal_hash(market_date: str, target_weights: pd.Series) -> str:
    payload = {"market_date": market_date, "weights_pct": format_weights_pct(target_weights)}
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def write_payload_if_requested(payload: dict, output_json: Optional[str]) -> None:
    if not output_json:
        return
    write_json_file(output_json, payload)


def main() -> None:
    print("Starting Dhan Cloud v21 mom10 x relative-volume weekly live + paper strategy execution...")

    try:
        client = build_client()
        prices, volume, metadata = fetch_universe_history(client, LOOKBACK_DAYS, END_DATE_OVERRIDE)
        if len(prices) <= WARMUP_DAYS:
            raise RuntimeError(
                f"Need more history. Got {len(prices)} rows but require more than {WARMUP_DAYS} to build the weekly paper signal."
            )

        scores = build_mom10_x_vr20_scores(prices, volume)
        raw_targets = build_top_n_targets(scores, top_n=2)
        smoothed_weights = apply_rebalance_policy(
            raw_targets,
            frequency="WEEKLY",
            trade_band=0.20,
            trade_step=0.50,
        )

        latest_market_date = metadata["latest_market_date"]
        latest_prices = {symbol: float(prices.iloc[-1][asset]) for asset, symbol in ASSET_TO_SYMBOL.items()}
        latest_raw_target = raw_targets.iloc[-1]
        latest_live_target = smoothed_weights.iloc[-1]
        latest_scores = scores.iloc[-1].sort_values(ascending=False)
        this_signal_hash = signal_hash(latest_market_date, latest_live_target)

        state_path = STATE_JSON_PATH
        journal_path = PAPER_JOURNAL_PATH
        order_journal_path = ORDER_JOURNAL_PATH
        state = load_state(state_path, INITIAL_CASH_INR)
        state_dirty = False

        nav_before = float(state["cash_inr"]) + holdings_value_inr(state["holdings_qty"], latest_prices)
        book_before_pct = book_weights_pct(state["cash_inr"], state["holdings_qty"], latest_prices)

        already_processed = (
            state.get("last_processed_market_date") == latest_market_date
            and state.get("last_signal_hash") == this_signal_hash
        )

        if already_processed:
            rebalance = {
                "cash_inr": round(float(state["cash_inr"]), 2),
                "holdings_qty": dict(state["holdings_qty"]),
                "nav_before_inr": round(nav_before, 2),
                "nav_after_inr": round(nav_before, 2),
                "target_qty": target_qty_from_weights(latest_live_target, nav_before, latest_prices),
                "planned_orders": [],
                "fills": [],
                "total_cost_inr": 0.0,
            }
            paper_status = "already_processed_for_latest_market_date"
        else:
            rebalance = execute_paper_rebalance(state, latest_live_target, latest_prices)
            state["cash_inr"] = rebalance["cash_inr"]
            state["holdings_qty"] = rebalance["holdings_qty"]
            state["last_processed_market_date"] = latest_market_date
            state["last_signal_hash"] = this_signal_hash
            state["last_nav_inr"] = rebalance["nav_after_inr"]
            state["runs"] = int(state.get("runs", 0)) + 1
            state_dirty = True
            append_journal(
                journal_path,
                {
                    "timestamp": datetime.now().isoformat(),
                    "model_name": MODEL_NAME,
                    "market_date": latest_market_date,
                    "signal_hash": this_signal_hash,
                    "nav_before_inr": rebalance["nav_before_inr"],
                    "nav_after_inr": rebalance["nav_after_inr"],
                    "cash_inr": rebalance["cash_inr"],
                    "target_weights_pct": format_weights_pct(latest_live_target),
                    "book_weights_pct": book_weights_pct(rebalance["cash_inr"], rebalance["holdings_qty"], latest_prices),
                    "fills": rebalance["fills"],
                    "total_cost_inr": rebalance["total_cost_inr"],
                },
            )
            paper_status = "processed"

        if state_dirty:
            save_state(state_path, state)
            state_dirty = False

        holdings_response = client.get_holdings()
        funds_response = client.get_fund_limits()
        live_holdings_qty = normalize_holdings(holdings_response)
        live_cash = available_cash_inr(funds_response)
        live_plan = build_live_rebalance_plan(
            latest_live_target,
            live_holdings_qty,
            live_cash,
            latest_prices,
            cash_buffer_pct=CASH_BUFFER_PCT,
            min_order_value=MIN_ORDER_VALUE,
        )
        live_book_before_pct = book_weights_pct(live_cash, live_holdings_qty, latest_prices)
        live_already_processed = (
            PLACE_ORDERS
            and state.get("last_live_processed_market_date") == latest_market_date
            and state.get("last_live_signal_hash") == this_signal_hash
        )
        live_execution_id = f"live_{int(time.time())}"
        if live_already_processed:
            live_order_status = "already_processed_for_latest_market_date"
            order_results = []
        else:
            order_results = place_live_orders(
                client,
                live_plan["orders"],
                place_orders=PLACE_ORDERS,
                after_market_order=AFTER_MARKET_ORDER,
                correlation_prefix=live_execution_id,
                max_order_count=MAX_ORDER_COUNT,
            )
            live_order_status = "submitted" if PLACE_ORDERS else "dry_run"
            if PLACE_ORDERS:
                state["last_live_processed_market_date"] = latest_market_date
                state["last_live_signal_hash"] = this_signal_hash
                state["last_live_execution_id"] = live_execution_id
                state_dirty = True
                append_journal(
                    order_journal_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "model_name": MODEL_NAME,
                        "market_date": latest_market_date,
                        "signal_hash": this_signal_hash,
                        "execution_id": live_execution_id,
                        "target_weights_pct": format_weights_pct(latest_live_target),
                        "live_plan": live_plan,
                        "order_results": order_results,
                    },
                )

        if state_dirty:
            save_state(state_path, state)

        nav_after = rebalance["nav_after_inr"]
        total_return_pct = ((nav_after / float(state["initial_cash_inr"])) - 1.0) * 100.0 if float(state["initial_cash_inr"]) > 0 else 0.0

        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "v21 mom10 x relative-volume weekly top-2 live plan and paper ledger generated from Dhan daily candles",
            "execution_id": f"exec_{int(time.time())}",
            "model_name": MODEL_NAME,
            "latest_market_date": latest_market_date,
            "data_source": metadata.get("data_source"),
            "requested_window": {
                "start": metadata.get("requested_start_date"),
                "end": metadata.get("requested_end_date"),
                "rows": metadata.get("rows"),
            },
            "order_mode": "live" if PLACE_ORDERS else "dry_run",
            "after_market_order": bool(AFTER_MARKET_ORDER),
            "paper_status": paper_status,
            "live_order_status": live_order_status,
            "state_path": state_path,
            "journal_path": journal_path,
            "order_journal_path": order_journal_path,
            "raw_top2_target_weights_pct": format_weights_pct(latest_raw_target),
            "smoothed_live_target_weights_pct": format_weights_pct(latest_live_target),
            "top_score_assets": {asset: round(float(value), 4) for asset, value in latest_scores.head(4).items()},
            "reference_prices_inr": {symbol: round(price, 4) for symbol, price in latest_prices.items()},
            "live_book": {
                "available_cash_inr": round(live_cash, 2),
                "book_weights_before_pct": live_book_before_pct,
                "holdings": holdings_snapshot(live_holdings_qty, latest_prices),
            },
            "live_rebalance": {
                "plan": live_plan,
                "order_results": order_results,
            },
            "paper_book": {
                "nav_before_inr": round(nav_before, 2),
                "nav_after_inr": round(nav_after, 2),
                "cash_inr": round(rebalance["cash_inr"], 2),
                "total_return_pct_since_inception": round(total_return_pct, 2),
                "book_weights_before_pct": book_before_pct,
                "book_weights_after_pct": book_weights_pct(rebalance["cash_inr"], rebalance["holdings_qty"], latest_prices),
                "holdings": holdings_snapshot(rebalance["holdings_qty"], latest_prices),
            },
            "paper_rebalance": {
                "target_qty": rebalance["target_qty"],
                "planned_orders": rebalance["planned_orders"],
                "fills": rebalance["fills"],
                "total_cost_inr": rebalance["total_cost_inr"],
            },
            "strategy_notes": [
                "This is the slowed weekly v21 challenger: top 2 assets by 10-day momentum times 20-day relative volume.",
                "Weekly schedule uses a 20% trade band and moves 50% toward the new target when the band is crossed.",
                "Live broker planning uses actual Dhan holdings and available cash, then optionally submits CNC market or AMO market orders.",
                "Paper fills use latest Dhan daily close and include India delivery-style transaction costs.",
                "CASH in the model maps to LIQUIDBEES in the paper ledger.",
                "Same-day reruns are idempotent when the market date and signal hash have not changed for either the paper ledger or the live order submission path.",
                "Dhan holdings sells may require eDIS or TPIN authorization depending on broker-side state.",
                "Sizing uses latest Dhan daily close as the reference price, not live LTP.",
            ],
        }

        print(f"Result: {json.dumps(payload, indent=2)}")
        write_payload_if_requested(payload, OUTPUT_JSON_PATH)

    except Exception as exc:
        print(f"Error in strategy execution: {exc}")
        raise

    print("Dhan Cloud v21 mom10 x relative-volume weekly live + paper strategy completed successfully")


if __name__ == "__main__":
    main()
