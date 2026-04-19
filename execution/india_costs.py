from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


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

    @property
    def bps(self) -> float:
        return self.total / self.notional * 10_000.0 if self.notional > 0.0 else 0.0


@dataclass(frozen=True)
class IndianDeliveryCostModel:
    """
    Official-style Indian delivery equity/ETF cost estimator.

    This mirrors the public Groww-style delivery cost stack we use for
    research and planning. It is still an estimator: live API charge previews
    remain the authority when an order is about to be submitted.
    """

    brokerage_cap: float = 20.0
    brokerage_rate: float = 0.001
    stt_rate: float = 0.001
    stamp_duty_buy_rate: float = 0.00015
    exchange_transaction_rate: float = 0.0000297
    sebi_turnover_rate: float = 0.000001
    ipft_rate: float = 0.000001
    dp_charge_base: float = 20.0
    dp_charge_threshold: float = 100.0
    gst_rate: float = 0.18
    charge_cash_proxy: bool = True

    def order_cost(self, notional: float, side: str) -> DeliveryCostBreakdown:
        value = max(float(notional), 0.0)
        normalized_side = side.upper()
        if value <= 0.0:
            return DeliveryCostBreakdown(0.0, normalized_side, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side for delivery cost: {side}")

        brokerage = min(self.brokerage_cap, value * self.brokerage_rate)
        stt = value * self.stt_rate
        stamp = value * self.stamp_duty_buy_rate if normalized_side == "BUY" else 0.0
        exchange = value * self.exchange_transaction_rate
        sebi = value * self.sebi_turnover_rate
        ipft = value * self.ipft_rate
        dp_base = self.dp_charge_base if normalized_side == "SELL" and value >= self.dp_charge_threshold else 0.0

        # GST is charged on broker/exchange-style charges, not on STT, stamp
        # duty, or pure market value. DP charge is also GST-able.
        gst = (brokerage + exchange + sebi + ipft + dp_base) * self.gst_rate
        return DeliveryCostBreakdown(
            notional=value,
            side=normalized_side,
            brokerage=brokerage,
            stt=stt,
            stamp_duty=stamp,
            exchange_transaction=exchange,
            sebi_turnover=sebi,
            ipft=ipft,
            dp_charge=dp_base,
            gst=gst,
        )

    def rebalance_cost(
        self,
        previous_weights: Mapping[str, float] | pd.Series,
        target_weights: Mapping[str, float] | pd.Series,
        portfolio_value: float,
    ) -> float:
        previous = pd.Series(previous_weights, dtype=float).fillna(0.0)
        target = pd.Series(target_weights, dtype=float).fillna(0.0)
        assets = previous.index.union(target.index)
        previous = previous.reindex(assets, fill_value=0.0)
        target = target.reindex(assets, fill_value=0.0)

        total = 0.0
        for asset, delta_weight in (target - previous).items():
            if asset == "CASH" and not self.charge_cash_proxy:
                continue
            notional = abs(float(delta_weight)) * max(float(portfolio_value), 0.0)
            if notional <= 0.0:
                continue
            side = "BUY" if delta_weight > 0 else "SELL"
            total += self.order_cost(notional, side).total
        return float(total)


DEFAULT_INDIAN_DELIVERY_COST_MODEL = IndianDeliveryCostModel()


def resolve_cost_model(name: str | None) -> IndianDeliveryCostModel | None:
    if name is None or name == "flat":
        return None
    if name in {"india_delivery", "official", "groww_delivery"}:
        return DEFAULT_INDIAN_DELIVERY_COST_MODEL
    raise ValueError(f"Unsupported cost model: {name}")
