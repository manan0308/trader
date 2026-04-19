from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from strategy.v9_engine import ALL
from broker.groww_client import DEFAULT_GROWW_UNIVERSE, GrowwInstrument
from execution.india_costs import DEFAULT_INDIAN_DELIVERY_COST_MODEL
from runtime.display_labels import asset_label


@dataclass(frozen=True)
class ExecutionConfig:
    min_order_value: float = 2_000.0
    min_trade_weight: float = 0.0025
    reserve_cash_value: float = 1_000.0
    reserve_cash_weight: float = 0.0025
    price_buffer_bps: float = 25.0
    estimated_cost_bps: float = 30.0
    cost_model: str = "india_delivery"
    default_lot_size: int = 1
    buy_priority: str = "largest_delta"
    use_cash_proxy: bool = True


@dataclass(frozen=True)
class PositionSnapshot:
    asset: str
    quantity: int
    price: float
    market_value: float


@dataclass(frozen=True)
class OrderIntent:
    asset: str
    side: str
    quantity: int
    reference_price: float
    order_price: float
    current_quantity: int
    target_quantity: int
    current_value: float
    target_value: float
    delta_value: float
    estimated_cost: float
    notes: str = ""


@dataclass(frozen=True)
class RebalancePlan:
    as_of: pd.Timestamp
    total_equity: float
    investable_equity: float
    current_cash: float
    reserve_cash: float
    starting_weights: pd.Series
    target_weights: pd.Series
    target_values: pd.Series
    current_values: pd.Series
    target_quantities: pd.Series
    post_trade_quantities: pd.Series
    post_trade_cash: float
    turnover_value: float
    orders: list[OrderIntent] = field(default_factory=list)


def _normalize_weights(target_weights: Mapping[str, float]) -> pd.Series:
    weights = pd.Series(0.0, index=ALL, dtype=float)
    for asset, value in target_weights.items():
        if asset in weights.index:
            weights[asset] = float(value)
    weights = weights.clip(lower=0.0)
    total = float(weights.sum())
    if total <= 0:
        weights["CASH"] = 1.0
        return weights
    return weights / total


def _coerce_prices(prices: Mapping[str, float]) -> pd.Series:
    series = pd.Series({asset: float(prices.get(asset, np.nan)) for asset in ALL}, dtype=float)
    if series.isna().any():
        missing = series[series.isna()].index.tolist()
        raise ValueError(f"Missing prices for assets: {missing}")
    # Reject non-positive prices up-front. A zero or negative fallback from a
    # stale quote cache would otherwise propagate through to the quantity
    # calculation (``target_value / price``) and produce ``inf`` shares, which
    # the downstream planner will happily round to a huge integer order.
    nonpositive = series[series <= 0.0]
    if not nonpositive.empty:
        raise ValueError(
            "Non-positive prices are not tradable: "
            f"{nonpositive.to_dict()}"
        )
    return series


def _coerce_quantities(quantities: Mapping[str, float | int]) -> pd.Series:
    series = pd.Series(0, index=ALL, dtype=int)
    for asset, value in quantities.items():
        if asset in series.index:
            series[asset] = max(int(value), 0)
    return series


def _lot_sizes(
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
    lot_size_map: Optional[Mapping[str, int]] = None,
    default_lot_size: int = 1,
) -> pd.Series:
    lot_sizes = pd.Series(default_lot_size, index=ALL, dtype=int)
    if lot_size_map:
        for asset, value in lot_size_map.items():
            if asset in lot_sizes.index:
                lot_sizes[asset] = max(int(value), 1)
    if instrument_map:
        for asset in instrument_map:
            if asset in lot_sizes.index:
                lot_sizes[asset] = max(int(lot_sizes[asset]), 1)
    return lot_sizes


def current_portfolio_snapshot(
    holdings: Mapping[str, float | int],
    prices: Mapping[str, float],
    available_cash: float,
) -> Dict[str, object]:
    px = _coerce_prices(prices)
    qty = _coerce_quantities(holdings)
    values = qty.astype(float) * px
    equity = float(values.sum() + available_cash)
    if equity <= 0:
        raise ValueError("Total equity must be positive.")
    weights = values / equity
    idle_cash_weight = float(available_cash) / equity
    positions = {
        asset: PositionSnapshot(
            asset=asset,
            quantity=int(qty[asset]),
            price=float(px[asset]),
            market_value=float(values[asset]),
        )
        for asset in ALL
    }
    return {
        "positions": positions,
        "prices": px,
        "quantities": qty,
        "values": values,
        "weights": weights,
        "equity": equity,
        "available_cash": float(available_cash),
        "idle_cash_weight": idle_cash_weight,
    }


def _round_down(quantity: float, lot_size: int) -> int:
    return max(int(np.floor(quantity / lot_size)) * lot_size, 0)


def _build_order(
    asset: str,
    side: str,
    quantity: int,
    prices: pd.Series,
    current_qty: pd.Series,
    target_qty: pd.Series,
    current_values: pd.Series,
    target_values: pd.Series,
    config: ExecutionConfig,
) -> OrderIntent:
    reference_price = float(prices[asset])
    buffer = config.price_buffer_bps / 10_000.0
    order_price = reference_price * (1.0 + buffer) if side == "BUY" else reference_price * (1.0 - buffer)
    delta_value = abs(quantity * reference_price)
    estimated_cost = _estimated_order_cost(delta_value, side, config)
    notes = "LIQUIDBEES cash-proxy" if asset == "CASH" else "risky-etf"
    return OrderIntent(
        asset=asset,
        side=side,
        quantity=int(quantity),
        reference_price=reference_price,
        order_price=order_price,
        current_quantity=int(current_qty[asset]),
        target_quantity=int(target_qty[asset]),
        current_value=float(current_values[asset]),
        target_value=float(target_values[asset]),
        delta_value=float(delta_value),
        estimated_cost=float(estimated_cost),
        notes=notes,
    )


def _estimated_order_cost(notional: float, side: str, config: ExecutionConfig) -> float:
    if config.cost_model == "india_delivery":
        return DEFAULT_INDIAN_DELIVERY_COST_MODEL.order_cost(notional, side).total
    if config.cost_model == "flat":
        return float(notional) * config.estimated_cost_bps / 10_000.0
    raise ValueError(f"Unsupported execution cost model: {config.cost_model}")


def _max_affordable_buy_quantity(
    max_quantity: int,
    price: float,
    available_cash: float,
    lot_size: int,
    config: ExecutionConfig,
) -> int:
    lot_size = max(int(lot_size), 1)
    max_units = max(int(max_quantity) // lot_size, 0)
    if max_units <= 0:
        return 0

    low = 0
    high = max_units
    while low < high:
        mid = (low + high + 1) // 2
        quantity = mid * lot_size
        gross = quantity * float(price)
        total_cost = gross + _estimated_order_cost(gross, "BUY", config)
        if total_cost <= available_cash + 1e-9:
            low = mid
        else:
            high = mid - 1
    return low * lot_size


def plan_rebalance(
    target_weights: Mapping[str, float],
    holdings: Mapping[str, float | int],
    prices: Mapping[str, float],
    available_cash: float,
    config: Optional[ExecutionConfig] = None,
    lot_size_map: Optional[Mapping[str, int]] = None,
    instrument_map: Optional[Mapping[str, GrowwInstrument]] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> RebalancePlan:
    cfg = config or ExecutionConfig()
    target_w = _normalize_weights(target_weights)
    snapshot = current_portfolio_snapshot(holdings, prices, available_cash)
    px: pd.Series = snapshot["prices"]  # type: ignore[assignment]
    current_qty: pd.Series = snapshot["quantities"]  # type: ignore[assignment]
    current_values: pd.Series = snapshot["values"]  # type: ignore[assignment]
    start_weights: pd.Series = snapshot["weights"]  # type: ignore[assignment]
    total_equity = float(snapshot["equity"])  # type: ignore[assignment]
    free_cash = float(snapshot["available_cash"])  # type: ignore[assignment]

    lot_sizes = _lot_sizes(
        instrument_map=instrument_map or DEFAULT_GROWW_UNIVERSE,
        lot_size_map=lot_size_map,
        default_lot_size=cfg.default_lot_size,
    )
    reserve_cash = max(cfg.reserve_cash_value, total_equity * cfg.reserve_cash_weight)
    reserve_cash = min(reserve_cash, total_equity)
    investable_equity = max(total_equity - reserve_cash, 0.0)

    target_values = target_w * investable_equity
    if not cfg.use_cash_proxy:
        target_values["CASH"] = 0.0

    raw_target_qty = pd.Series(0, index=ALL, dtype=int)
    for asset in ALL:
        if asset == "CASH" and not cfg.use_cash_proxy:
            raw_target_qty[asset] = 0
            continue
        raw_target_qty[asset] = _round_down(float(target_values[asset] / px[asset]), int(lot_sizes[asset]))

    delta_weights = (target_w - start_weights).abs()
    working_qty = current_qty.copy()
    working_cash = free_cash
    turnover_value = 0.0
    planned_orders: list[OrderIntent] = []

    sell_assets: list[tuple[str, int, float]] = []
    buy_assets: list[tuple[str, int, float]] = []
    for asset in ALL:
        if asset == "CASH" and not cfg.use_cash_proxy:
            continue
        delta_qty = int(raw_target_qty[asset] - working_qty[asset])
        notional = abs(delta_qty) * float(px[asset])
        if delta_qty < 0:
            sell_assets.append((asset, delta_qty, notional))
        elif delta_qty > 0:
            buy_assets.append((asset, delta_qty, notional))

    sell_assets.sort(key=lambda item: item[2], reverse=True)
    for asset, delta_qty, notional in sell_assets:
        if notional < cfg.min_order_value and float(delta_weights[asset]) < cfg.min_trade_weight:
            continue
        quantity = min(abs(delta_qty), int(working_qty[asset]))
        if quantity <= 0:
            continue
        gross_proceeds = quantity * float(px[asset])
        fees = _estimated_order_cost(gross_proceeds, "SELL", cfg)
        working_qty[asset] -= quantity
        working_cash += gross_proceeds - fees
        turnover_value += gross_proceeds
        planned_orders.append(
            _build_order(
                asset=asset,
                side="SELL",
                quantity=quantity,
                prices=px,
                current_qty=current_qty,
                target_qty=raw_target_qty,
                current_values=current_values,
                target_values=target_values,
                config=cfg,
            )
        )

    if cfg.buy_priority == "largest_delta":
        buy_assets.sort(key=lambda item: item[2], reverse=True)
    elif cfg.buy_priority == "small_first":
        buy_assets.sort(key=lambda item: item[2])

    for asset, delta_qty, notional in buy_assets:
        if notional < cfg.min_order_value and float(delta_weights[asset]) < cfg.min_trade_weight:
            continue
        quantity = min(
            int(delta_qty),
            _max_affordable_buy_quantity(
                max_quantity=int(delta_qty),
                price=float(px[asset]),
                available_cash=working_cash,
                lot_size=int(lot_sizes[asset]),
                config=cfg,
            ),
        )
        if quantity <= 0:
            continue
        gross_spend = quantity * float(px[asset])
        fees = _estimated_order_cost(gross_spend, "BUY", cfg)
        working_qty[asset] += quantity
        working_cash -= gross_spend + fees
        turnover_value += gross_spend
        planned_orders.append(
            _build_order(
                asset=asset,
                side="BUY",
                quantity=quantity,
                prices=px,
                current_qty=current_qty,
                target_qty=raw_target_qty,
                current_values=current_values,
                target_values=target_values,
                config=cfg,
            )
        )

    return RebalancePlan(
        as_of=as_of or pd.Timestamp.utcnow().tz_localize(None),
        total_equity=total_equity,
        investable_equity=investable_equity,
        current_cash=free_cash,
        reserve_cash=reserve_cash,
        starting_weights=start_weights,
        target_weights=target_w,
        target_values=target_values,
        current_values=current_values,
        target_quantities=raw_target_qty,
        post_trade_quantities=working_qty,
        post_trade_cash=working_cash,
        turnover_value=turnover_value,
        orders=planned_orders,
    )


def format_plan(plan: RebalancePlan) -> str:
    lines = [
        f"As of: {plan.as_of:%Y-%m-%d %H:%M:%S}",
        f"Total equity: {plan.total_equity:,.2f}",
        f"Current cash: {plan.current_cash:,.2f}",
        f"Reserve cash: {plan.reserve_cash:,.2f}",
        f"Investable equity: {plan.investable_equity:,.2f}",
        f"Post-trade cash: {plan.post_trade_cash:,.2f}",
        f"Orders: {len(plan.orders)}",
    ]
    for order in plan.orders:
        lines.append(
            f"  {order.side:<4} {asset_label(order.asset):<10} qty {order.quantity:>6} "
            f"ref {order.reference_price:>9.2f} "
            f"delta {order.delta_value:>10.2f} "
            f"cost {order.estimated_cost:>8.2f}"
        )
    return "\n".join(lines)
