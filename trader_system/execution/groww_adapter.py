from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from trader_system.strategy.v9_engine import ALL
from trader_system.execution.rebalance_core import OrderIntent
from trader_system.broker.groww_client import DEFAULT_GROWW_UNIVERSE, GrowwInstrument, GrowwSession


@dataclass(frozen=True)
class GrowwOrderRequest:
    asset: str
    exchange: str
    segment: str
    trading_symbol: str
    side: str
    quantity: int
    order_type: str
    product: str
    validity: str
    limit_price: Optional[float]
    reference_price: float


def extract_asset_quantities(
    holdings_payload: object,
    positions_payload: object,
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
) -> Dict[str, int]:
    mapping = instrument_map or DEFAULT_GROWW_UNIVERSE
    symbol_to_asset = {
        instrument.trading_symbol.upper(): asset
        for asset, instrument in mapping.items()
    }
    quantities = {asset: 0 for asset in mapping}

    def update_from_rows(rows: object) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = (
                row.get("tradingSymbol")
                or row.get("tradingsymbol")
                or row.get("symbol")
                or row.get("exchangeTradedSymbol")
                or row.get("exchange_trading_symbol")
            )
            if not symbol:
                continue
            asset = symbol_to_asset.get(str(symbol).upper())
            if not asset:
                continue
            quantity = (
                row.get("quantity")
                or row.get("availableQuantity")
                or row.get("sellableQuantity")
                or row.get("netQuantity")
                or row.get("netQty")
                or 0
            )
            try:
                quantities[asset] = max(quantities[asset], int(float(quantity)))
            except (TypeError, ValueError):
                continue

    if isinstance(holdings_payload, dict):
        update_from_rows(holdings_payload.get("holdings"))
    if isinstance(positions_payload, dict):
        update_from_rows(positions_payload.get("positions"))
    return quantities


def groww_cash_from_funds_payload(funds_payload: object) -> Optional[float]:
    if not isinstance(funds_payload, dict):
        return None
    candidates = [
        funds_payload.get("availableCash"),
        funds_payload.get("availableBalance"),
        funds_payload.get("cashAvailable"),
        funds_payload.get("available", {}).get("cash") if isinstance(funds_payload.get("available"), dict) else None,
        funds_payload.get("equity", {}).get("available") if isinstance(funds_payload.get("equity"), dict) else None,
    ]
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def current_holdings_snapshot(
    session: GrowwSession,
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
) -> Dict[str, object]:
    mapping = instrument_map or session.instrument_map
    holdings = session.client.get_holdings_for_user()
    positions = session.client.get_positions_for_user(segment=session.client.SEGMENT_CASH)
    margins = session.client.get_available_margin_details()
    quantities = extract_asset_quantities(holdings, positions, instrument_map=mapping)
    return {
        "holdings_payload": holdings,
        "positions_payload": positions,
        "margins_payload": margins,
        "quantities": quantities,
        "available_cash": groww_cash_from_funds_payload(margins),
    }


def build_groww_order_request(
    order: OrderIntent,
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
    order_type: str = "LIMIT",
    product: str = "CNC",
    validity: str = "DAY",
) -> GrowwOrderRequest:
    mapping = instrument_map or DEFAULT_GROWW_UNIVERSE
    instrument = mapping[order.asset]
    limit_price = order.order_price if order_type.upper() == "LIMIT" else None
    return GrowwOrderRequest(
        asset=order.asset,
        exchange=instrument.exchange,
        segment=instrument.segment,
        trading_symbol=instrument.trading_symbol,
        side=order.side,
        quantity=order.quantity,
        order_type=order_type.upper(),
        product=product.upper(),
        validity=validity.upper(),
        limit_price=limit_price,
        reference_price=order.reference_price,
    )


def build_groww_order_requests(
    orders: Iterable[OrderIntent],
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
    order_type: str = "LIMIT",
    product: str = "CNC",
    validity: str = "DAY",
) -> List[GrowwOrderRequest]:
    return [
        build_groww_order_request(
            order,
            instrument_map=instrument_map,
            order_type=order_type,
            product=product,
            validity=validity,
        )
        for order in orders
    ]


def order_request_to_payload(order: GrowwOrderRequest) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "trading_symbol": order.trading_symbol,
        "exchange": order.exchange,
        "segment": order.segment,
        "transaction_type": order.side,
        "quantity": order.quantity,
        "order_type": order.order_type,
        "product": order.product,
        "validity": order.validity,
    }
    if order.limit_price is not None:
        payload["price"] = round(float(order.limit_price), 2)
    return payload


def groww_price_map(
    session: GrowwSession,
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
    fallback_prices: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    mapping = instrument_map or session.instrument_map
    fallback = {asset: float(price) for asset, price in (fallback_prices or {}).items()}
    response = session.multi_ltp(mapping.keys())

    out: Dict[str, float] = {}
    for asset, instrument in mapping.items():
        candidates = [
            f"{instrument.exchange}_{instrument.trading_symbol}",
            instrument.groww_symbol,
            instrument.trading_symbol,
        ]
        chosen = None
        for key in candidates:
            if key not in response:
                continue
            try:
                chosen = float(response[key])
                break
            except (TypeError, ValueError):
                continue
        if chosen is None:
            chosen = float(fallback.get(asset, 0.0))
        out[asset] = chosen
    return out


def actual_weight_snapshot(
    quantities: Mapping[str, int],
    price_map: Mapping[str, float],
    available_cash: float,
) -> Dict[str, object]:
    quantity_series = pd.Series({asset: int(quantities.get(asset, 0)) for asset in ALL}, dtype=float)
    price_series = pd.Series({asset: float(price_map.get(asset, 0.0)) for asset in ALL}, dtype=float)
    values = quantity_series * price_series
    holdings_value = float(values.sum())
    total_equity = holdings_value + float(available_cash)
    if total_equity <= 0:
        total_equity = 1.0
    weights = {asset: float(values[asset] / total_equity) for asset in ALL}
    weights["CASH"] = float(available_cash) / total_equity
    return {
        "total_equity": total_equity,
        "holdings_value": holdings_value,
        "available_cash": float(available_cash),
        "quantities": {asset: int(quantity_series[asset]) for asset in ALL},
        "market_values": {asset: float(values[asset]) for asset in ALL},
        "weights": weights,
    }
