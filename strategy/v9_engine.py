#!/usr/bin/env python3
"""
ALPHA ENGINE v9 — LOW-TURNOVER CORE + TACTICAL TILT
===================================================

This version intentionally removes the parts of v8 that were statistically
fragile at daily frequency:

1. No daily Kelly sizing.
2. No Markov regime switching.
3. No cost-unaware daily rebalancing.

What replaces them:

1. A diversified risk-weighted core that keeps the universe alpha.
2. A small tactical sleeve that tilts toward slow, cross-asset winners.
3. Daily monitoring for breadth deterioration and crash conditions.
4. Weekly or monthly execution with bands and partial fills.

Design goals:
- Keep the 1-day execution lag.
- Keep realistic 30 bps trading costs.
- Trade rarely enough that costs stop dominating outcomes.
- Stay simple enough that walk-forward results mean something.

Future hook:
- `risk_off_override` lets an external LLM layer reduce the risk budget.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from broker.groww_client import GROWW_MIN_HISTORY_START, GrowwSession, load_groww_universe
from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
from market_data.market_store import load_processed_matrix


RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
ALL = RISKY + ["CASH"]

DEFAULT_START = "2012-01-01"
DEFAULT_BACKTEST_START = "2012-04-01"
DEFAULT_BACKTEST_END = "2026-03-31"
DEFAULT_RF = 0.065
DEFAULT_TX = 0.003
WARMUP_DAYS = 252
TRAIN_DAYS = 504
TEST_DAYS = 126

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
CONFIG_DATA_DIR = BASE_DIR / "config" / "data"
INDIA_SILVER_OVERRIDE_PATH = CONFIG_DATA_DIR / "silver_india_daily.csv"
PRICE_CACHE = CACHE_DIR / "alpha_v9_prices.csv"
RESULTS_CACHE = CACHE_DIR / "alpha_v9_results.json"


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


@dataclass(frozen=True)
class NarrativeOverlay:
    risk_off: pd.Series
    asset_bias: pd.DataFrame


class PriceSource:
    def fetch(self, start: str, refresh: bool = False) -> pd.DataFrame:
        raise NotImplementedError


class YahooFinanceSource(PriceSource):
    RESEARCH_SYMBOLS = {
        "NIFTY": "^NSEI",
        "MIDCAP": "^NSEMDCP50",
        "SMALLCAP": "^CRSMID",
        "GOLD": "GOLDBEES.NS",
        "US": "MON100.NS",
        "CASH": "LIQUIDBEES.NS",
        "SI": "SI=F",
        "FX": "INR=X",
    }
    BENCHMARK_SYMBOLS = {
        "NIFTY": "^NSEI",
        "MIDCAP": "^NSEMDCP50",
        "SMALLCAP": "^CRSMID",
        "GOLD": "GOLDBEES.NS",
        "US": "MON100.NS",
        "SI": "SI=F",
        "FX": "INR=X",
    }
    TRADABLE_PROXY_SYMBOLS = {
        "NIFTY": "NIFTYBEES.NS",
        "MIDCAP": "MID150BEES.NS",
        "SMALLCAP": "SMALLCAP.NS",
        "GOLD": "GOLDBEES.NS",
        "SILVER": "SILVERBEES.NS",
        "US": "MON100.NS",
        "CASH": "LIQUIDBEES.NS",
    }

    def __init__(self, universe_mode: str = "research") -> None:
        if universe_mode not in {"research", "benchmark", "tradable"}:
            raise ValueError(f"unsupported universe mode: {universe_mode}")
        self.universe_mode = universe_mode
        self.cache_path = CACHE_DIR / f"alpha_v9_prices_{universe_mode}.csv"

    @property
    def symbols(self) -> Dict[str, str]:
        if self.universe_mode == "tradable":
            return self.TRADABLE_PROXY_SYMBOLS
        if self.universe_mode == "benchmark":
            return self.BENCHMARK_SYMBOLS
        return self.RESEARCH_SYMBOLS

    def _download_close(self, ticker: str, start: str, end: str | None = None) -> pd.Series:
        last_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                history = yf.Ticker(ticker).history(
                    start=start,
                    end=end,
                    auto_adjust=True,
                    repair=True,
                )
                if history.empty or "Close" not in history.columns:
                    raise RuntimeError(f"empty history for {ticker}")
                return clean_price_series(history["Close"])
            except Exception as exc:  # pragma: no cover - defensive path
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"failed to download {ticker}: {last_error}")

    def _slice_window(self, prices: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
        out = prices.sort_index()
        out = out.loc[pd.Timestamp(start) :]
        if end:
            out = out.loc[: pd.Timestamp(end)]
        return out

    def _download_prices(self, start: str, end: str | None = None) -> pd.DataFrame:
        raw: Dict[str, pd.Series] = {}
        for name, ticker in self.symbols.items():
            raw[name] = self._download_close(ticker, start, end)
        local_india_silver = load_local_price_override(INDIA_SILVER_OVERRIDE_PATH)

        anchor_name = "CASH" if "CASH" in raw else next(iter(raw))
        prices = pd.DataFrame(index=raw[anchor_name].index)
        if self.universe_mode == "tradable":
            for asset in ALL:
                prices[asset] = raw[asset]
        elif self.universe_mode == "benchmark":
            for asset in ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "US"]:
                prices[asset] = raw[asset]
            if local_india_silver is not None:
                prices["SILVER"] = local_india_silver
            else:
                common_silver = raw["SI"].index.intersection(raw["FX"].index)
                prices["SILVER"] = clean_price_series(raw["SI"].loc[common_silver] * raw["FX"].loc[common_silver])
        else:
            for asset in ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "US", "CASH"]:
                prices[asset] = raw[asset]
            if local_india_silver is not None:
                prices["SILVER"] = local_india_silver
            else:
                common_idx = raw["SI"].index.intersection(raw["FX"].index)
                prices["SILVER"] = clean_price_series(raw["SI"].loc[common_idx] * raw["FX"].loc[common_idx])
        if self.universe_mode == "benchmark":
            rf_daily = (1.0 + DEFAULT_RF) ** (1.0 / 252.0) - 1.0
            prices["CASH"] = 100.0 * np.power(1.0 + rf_daily, np.arange(len(prices), dtype=float))
        prices = prices.ffill(limit=5).dropna()
        return self._slice_window(prices[ALL], start, end)

    def fetch(self, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset_name = {
            "research": "india_research",
            "benchmark": "india_benchmark",
            "tradable": "india_tradable",
        }[self.universe_mode]

        if not refresh:
            local_matrix = load_processed_matrix(dataset_name, start=start, end=end)
            if local_matrix is not None and set(ALL).issubset(local_matrix.columns):
                return local_matrix[ALL]

        if self.cache_path.exists() and not refresh:
            cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
            if set(ALL).issubset(cached.columns):
                return self._slice_window(cached[ALL], start, end)

        try:
            prices = self._download_prices(start, end)
            prices.to_csv(self.cache_path, index=True)
            return prices
        except Exception:
            if self.cache_path.exists():
                cached = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                if set(ALL).issubset(cached.columns):
                    return self._slice_window(cached[ALL], start, end)
            raise


class GrowwSource(PriceSource):
    def __init__(self, session: GrowwSession) -> None:
        self.session = session

    def fetch(self, start: str, refresh: bool = False) -> pd.DataFrame:
        return self.session.fetch_universe_prices(start=start, refresh=refresh)


def clean_price_series(series: pd.Series) -> pd.Series:
    """
    Yahoo occasionally serves split/scale glitches for NSE ETFs.
    We remove impossible level shifts and impossible one-day jumps before
    constructing returns.
    """
    clean = pd.Series(series).copy().astype(float)
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean = clean.sort_index()

    rolling_median = clean.rolling(20, min_periods=5).median()
    level_bad = ((clean / rolling_median) > 4.0) | ((clean / rolling_median) < 0.25)
    clean[level_bad] = np.nan

    for _ in range(2):
        jumps = clean.pct_change(fill_method=None).abs() > 0.50
        clean[jumps] = np.nan

    return clean.ffill(limit=5)


def load_local_price_override(path: Path) -> pd.Series | None:
    if not path.exists():
        return None

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    if frame.empty:
        return None

    columns = {str(col).strip().lower(): str(col) for col in frame.columns}
    date_col = columns.get("date")
    if date_col is None:
        raise ValueError(f"{path} must contain a Date column.")

    value_col = None
    for candidate in ("close", "settle", "price", "value", "last"):
        if candidate in columns:
            value_col = columns[candidate]
            break
    if value_col is None:
        raise ValueError(f"{path} must contain one of Close/Settle/Price/Value/Last.")

    series = pd.Series(frame[value_col].astype(float).values, index=pd.to_datetime(frame[date_col]), name="Close")
    series = series.sort_index()
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    return clean_price_series(series)


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
    if not selected:
        return pd.Series(dtype=float)

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
    rs = up.rolling(14).mean() / down.rolling(14).mean().replace(0, np.nan)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))
    asset_vol20 = returns[RISKY].rolling(20).std() * np.sqrt(252)
    asset_vol252 = returns[RISKY].rolling(252).std() * np.sqrt(252)

    return {
        "returns": returns,
        "sma50": prices[RISKY].rolling(50).mean(),
        "sma200": prices[RISKY].rolling(200).mean(),
        "mom63": prices[RISKY].pct_change(63, fill_method=None),
        "mom126": prices[RISKY].pct_change(126, fill_method=None),
        "mom252": prices[RISKY].pct_change(252, fill_method=None),
        "vol63": returns[RISKY].rolling(63).std() * np.sqrt(252),
        "rsi14": rsi14,
        "asset_dd20": prices[RISKY] / prices[RISKY].rolling(20).max() - 1.0,
        "asset_vol_ratio": asset_vol20 / asset_vol252.replace(0, np.nan),
        "breakout252": prices[RISKY] >= prices[RISKY].rolling(252).max().shift(1),
        "cash126": prices["CASH"].pct_change(126, fill_method=None),
        "dd20": risky_nav / risky_nav.rolling(20).max() - 1.0,
        "dd63": risky_nav / risky_nav.rolling(63).max() - 1.0,
        "vol20": risky_eq.rolling(20).std() * np.sqrt(252),
        "vol252": risky_eq.rolling(252).std() * np.sqrt(252),
        "vol252_median": (risky_eq.rolling(252).std() * np.sqrt(252)).expanding().median(),
    }


def crash_signal(
    features: Dict[str, pd.DataFrame | pd.Series],
    dt: pd.Timestamp,
) -> bool:
    dd20 = float(features["dd20"].loc[dt])  # type: ignore[index]
    dd63 = float(features["dd63"].loc[dt])  # type: ignore[index]
    vol20 = float(features["vol20"].loc[dt])  # type: ignore[index]
    vol252 = float(features["vol252"].loc[dt])  # type: ignore[index]
    vol252_median = float(features["vol252_median"].loc[dt])  # type: ignore[index]

    fast_crash = dd20 < -0.12 and vol20 > vol252 * 1.35
    slow_crash = dd63 < -0.10 and vol20 > vol252_median * 1.25
    return bool(fast_crash or slow_crash)


def breadth_risk_scale(breadth: int) -> float:
    mapping = {
        0: 0.70,
        1: 0.80,
        2: 0.90,
        3: 0.95,
        4: 1.00,
        5: 1.00,
        6: 1.00,
    }
    return mapping.get(breadth, 1.00)


def run_strategy(
    prices: pd.DataFrame,
    config: StrategyConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    """
    Daily signal evaluation, lower-frequency execution:
    - Keep a broad inverse-vol core across all risky assets.
    - Add a small tactical tilt to the strongest trending assets.
    - Scale the whole risky bucket down only when breadth or crash evidence weakens.
    - Allow daily crash cuts between scheduled rebalances.
    """
    features = build_features(prices)
    schedule = schedule_flags(prices.index, config.execution_frequency)
    override = overlay.risk_off.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0) if overlay is not None else pd.Series(0.0, index=prices.index)
    asset_bias = overlay.asset_bias.reindex(prices.index).fillna(0.0) if overlay is not None else pd.DataFrame(0.0, index=prices.index, columns=RISKY)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = features["sma50"]  # type: ignore[assignment]
    sma200 = features["sma200"]  # type: ignore[assignment]
    mom63 = features["mom63"]  # type: ignore[assignment]
    mom126 = features["mom126"]  # type: ignore[assignment]
    mom252 = features["mom252"]  # type: ignore[assignment]
    vol63 = features["vol63"]  # type: ignore[assignment]
    rsi14 = features["rsi14"]  # type: ignore[assignment]
    asset_dd20 = features["asset_dd20"]  # type: ignore[assignment]
    asset_vol_ratio = features["asset_vol_ratio"]  # type: ignore[assignment]
    breakout252 = features["breakout252"]  # type: ignore[assignment]
    cash126 = features["cash126"]  # type: ignore[assignment]

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

            llm_override = float(override.loc[dt])
            risk_scale = min(risk_scale, 1.0 - 0.65 * llm_override)

            score = (
                0.20 * (mom63.loc[dt] / vol63.loc[dt].replace(0, np.nan))
                + 0.35 * mom126.loc[dt]
                + 0.45 * mom252.loc[dt]
            )
            score = score.replace([np.inf, -np.inf], np.nan).fillna(-999.0).sort_values(ascending=False)

            # Pattern lab repeatedly found that broad-risk assets recover well after
            # sharp pullbacks when the long-term trend survives, so give that setup a
            # modest bonus rather than turning the model into pure mean reversion.
            rebound_setup = trend & (asset_dd20.loc[dt] <= -0.08) & (rsi14.loc[dt] <= 45.0)
            score = score + rebound_setup.astype(float) * 0.04

            # Gold tends to be more useful after volatility shocks, and silver
            # breakouts have shown medium-horizon persistence in the pattern lab.
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

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            if target[RISKY].sum() > 0 and float(bias_row.abs().sum()) > 0:
                # Narrative bias should only tilt the risky sleeve, not replace it.
                multipliers = 1.0 + 0.15 * bias_row
                adjusted = target[RISKY] * multipliers
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum() * target[RISKY].sum()
                    target[RISKY] = adjusted

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

        llm_live_risk = float(override.loc[dt])
        if llm_live_risk > 0.15 and current[RISKY].sum() > 0:
            # Narrative event-risk overrides should be able to cut exposure
            # immediately instead of waiting for the normal rebalance band.
            min_cash = float(np.clip(0.20 + 0.45 * llm_live_risk, 0.20, 0.70))
            max_risky = 1.0 - min_cash
            risky_sum = float(current[RISKY].sum())
            if risky_sum > max_risky:
                current = current.copy()
                current[RISKY] *= max_risky / risky_sum
                current["CASH"] = 1.0 - current[RISKY].sum()

        weights.iloc[i] = current

    return weights


def portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    tx_cost: float,
    cost_model: Optional[IndianDeliveryCostModel] = None,
    base_value: float = 1_000_000.0,
) -> pd.Series:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    aligned_weights = weights.reindex(index=prices.index, columns=ALL).ffill().fillna(0.0)
    lagged = aligned_weights.shift(1).ffill().fillna(0.0)

    if cost_model is None:
        gross = (lagged * returns).sum(axis=1)
        turnover = aligned_weights.diff().abs().sum(axis=1) / 2.0
        net = gross - turnover * tx_cost
        return net.iloc[1:]

    equity = float(base_value)
    net_returns: list[float] = []
    net_index: list[pd.Timestamp] = []
    for i, dt in enumerate(prices.index):
        if i == 0:
            continue
        gross_return = float((lagged.loc[dt] * returns.loc[dt]).sum())
        equity_after_market = equity * (1.0 + gross_return)
        cost = cost_model.rebalance_cost(
            lagged.loc[dt],
            aligned_weights.loc[dt],
            equity_after_market,
        )
        net_equity = max(equity_after_market - cost, 0.0)
        net_returns.append(net_equity / equity - 1.0 if equity > 0.0 else 0.0)
        net_index.append(dt)
        equity = net_equity

    return pd.Series(net_returns, index=pd.DatetimeIndex(net_index), dtype=float)


def performance_metrics(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    label: str,
    rf: float,
    tx_cost: float,
    cost_model: Optional[IndianDeliveryCostModel] = None,
    base_value: float = 1_000_000.0,
) -> Dict[str, object]:
    daily = portfolio_returns(prices, weights, tx_cost=tx_cost, cost_model=cost_model, base_value=base_value)
    equity = (1.0 + daily).cumprod()
    years = len(daily) / 252

    rf_daily = (1.0 + rf) ** (1 / 252) - 1.0
    excess = daily - rf_daily
    vol = daily.std() * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1.0).min()
    turnover = (weights.fillna(0.0).diff().abs().sum(axis=1) / 2.0).sum() / years

    result = {
        "label": label,
        "weights": weights,
        "returns": daily,
        "equity": equity,
        "cagr": equity.iloc[-1] ** (1 / years) - 1.0,
        "vol": vol,
        "sharpe": excess.mean() * 252 / vol if vol > 0 else np.nan,
        "mdd": mdd,
        "calmar": (equity.iloc[-1] ** (1 / years) - 1.0) / abs(mdd) if mdd < 0 else np.nan,
        "turnover": turnover,
        "avg_cash": weights["CASH"].mean(),
    }
    return result


def benchmark_weights(prices: pd.DataFrame, kind: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    if kind == "Nifty B&H":
        weights["NIFTY"] = 1.0
    elif kind == "Smallcap B&H":
        weights["SMALLCAP"] = 1.0
    elif kind == "EqWt Risky":
        for asset in RISKY:
            weights[asset] = 1.0 / len(RISKY)
    elif kind == "60/40 Nifty/Cash":
        weights["NIFTY"] = 0.60
        weights["CASH"] = 0.40
    else:
        raise ValueError(f"unknown benchmark: {kind}")
    return weights


def _overlay_window(index: pd.DatetimeIndex, start: pd.Timestamp, holding_days: int) -> pd.DatetimeIndex:
    active = index[index >= start]
    if len(active) == 0:
        return active
    return active[: max(1, int(holding_days))]


def load_llm_overlay(path: Optional[str], index: pd.DatetimeIndex) -> Optional[NarrativeOverlay]:
    if not path:
        return None

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    risk_off = pd.Series(0.0, index=index)
    asset_bias = pd.DataFrame(0.0, index=index, columns=RISKY)

    def apply_overlay_item(item: Dict[str, object]) -> None:
        start = pd.Timestamp(item["date"])
        window = _overlay_window(index, start, int(item.get("holding_days", 1)))
        if len(window) == 0:
            return

        risk_value = float(item.get("risk_off_override", 0.0))
        risk_off.loc[window] = np.maximum(risk_off.loc[window], risk_value)

        bias_map = item.get("asset_bias", {})
        if isinstance(bias_map, dict):
            for asset in RISKY:
                if asset in bias_map:
                    asset_bias.loc[window, asset] = np.clip(float(bias_map[asset]), -1.0, 1.0)

    if isinstance(payload, dict):
        risk_off[:] = float(payload.get("default_risk_off_override", 0.0))

        if isinstance(payload.get("dates"), list):
            for item in payload["dates"]:
                apply_overlay_item(item)

        if isinstance(payload.get("overrides"), dict):
            for date_str, value in payload["overrides"].items():
                date = pd.Timestamp(date_str)
                window = _overlay_window(index, date, 1)
                if len(window) > 0:
                    risk_off.loc[window] = float(value)

    elif isinstance(payload, list):
        for item in payload:
            apply_overlay_item(item)

    else:
        raise ValueError("Unsupported LLM override file format.")

    return NarrativeOverlay(
        risk_off=risk_off.clip(lower=0.0, upper=1.0),
        asset_bias=asset_bias.clip(lower=-1.0, upper=1.0),
    )


def format_metrics(metrics: Dict[str, object]) -> Dict[str, str]:
    return {
        "CAGR": f"{metrics['cagr']:.1%}",
        "Vol": f"{metrics['vol']:.1%}",
        "Sharpe": f"{metrics['sharpe']:.2f}",
        "MaxDD": f"{metrics['mdd']:.1%}",
        "Calmar": f"{metrics['calmar']:.2f}",
        "AnnTurn": f"{metrics['turnover']:.0%}",
        "AvgCash": f"{metrics['avg_cash']:.0%}",
    }


def print_results_table(title: str, results: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    headers = ["Label", "CAGR", "Vol", "Sharpe", "MaxDD", "Calmar", "AnnTurn", "AvgCash"]
    widths = {"Label": 24, "CAGR": 9, "Vol": 9, "Sharpe": 8, "MaxDD": 9, "Calmar": 8, "AnnTurn": 9, "AvgCash": 9}
    header_row = " ".join(f"{col:>{widths[col]}}" for col in headers)
    print(header_row)
    print("-" * len(header_row))

    for result in results:
        fmt = format_metrics(result)
        row = [
            f"{str(result['label']):>{widths['Label']}}",
            f"{fmt['CAGR']:>{widths['CAGR']}}",
            f"{fmt['Vol']:>{widths['Vol']}}",
            f"{fmt['Sharpe']:>{widths['Sharpe']}}",
            f"{fmt['MaxDD']:>{widths['MaxDD']}}",
            f"{fmt['Calmar']:>{widths['Calmar']}}",
            f"{fmt['AnnTurn']:>{widths['AnnTurn']}}",
            f"{fmt['AvgCash']:>{widths['AvgCash']}}",
        ]
        print(" ".join(row))


def strategy_score(metrics: Dict[str, object]) -> float:
    sharpe = float(metrics["sharpe"])
    calmar = float(metrics["calmar"])
    turnover = float(metrics["turnover"])
    return sharpe + 0.35 * calmar - 0.10 * turnover


def walk_forward(
    prices: pd.DataFrame,
    candidates: List[StrategyConfig],
    rf: float,
    tx_cost: float,
    overlay: Optional[NarrativeOverlay] = None,
    cost_model: Optional[IndianDeliveryCostModel] = None,
    base_value: float = 1_000_000.0,
) -> Dict[str, object]:
    windows = []
    start = 0
    n = len(prices)
    while start + TRAIN_DAYS + TEST_DAYS <= n:
        windows.append((start, start + TRAIN_DAYS, start + TRAIN_DAYS + TEST_DAYS))
        start += TEST_DAYS

    if not windows:
        empty_returns = pd.Series(dtype=float)
        empty_weights = pd.DataFrame(columns=ALL, dtype=float)
        return {
            "windows": [],
            "returns": empty_returns,
            "metrics": {
                "label": "WFO Stitched OOS",
                "weights": empty_weights,
                "returns": empty_returns,
                "equity": pd.Series(dtype=float),
                "cagr": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "mdd": np.nan,
                "calmar": np.nan,
                "turnover": np.nan,
                "avg_cash": np.nan,
            },
        }

    oos_returns: List[pd.Series] = []
    oos_weights: List[pd.DataFrame] = []
    picked_rows: List[Dict[str, object]] = []

    for window_id, (train_start, train_end, test_end) in enumerate(windows, start=1):
        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[train_end:test_end]

        scored: List[tuple[float, StrategyConfig]] = []
        for config in candidates:
            train_overlay = None
            if overlay is not None:
                train_overlay = NarrativeOverlay(
                    risk_off=overlay.risk_off.loc[train_prices.index],
                    asset_bias=overlay.asset_bias.loc[train_prices.index],
                )
            train_weights = run_strategy(train_prices, config, overlay=train_overlay)
            train_metrics = performance_metrics(
                train_prices,
                train_weights,
                config.name,
                rf=rf,
                tx_cost=tx_cost,
                cost_model=cost_model,
                base_value=base_value,
            )
            scored.append((strategy_score(train_metrics), config))

        _, best = max(scored, key=lambda item: item[0])
        combined = prices.iloc[train_start:test_end]
        combined_overlay = None
        if overlay is not None:
            combined_overlay = NarrativeOverlay(
                risk_off=overlay.risk_off.loc[combined.index],
                asset_bias=overlay.asset_bias.loc[combined.index],
            )
        combined_weights = run_strategy(combined, best, overlay=combined_overlay)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(
            test_prices,
            test_weights,
            best.name,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )

        picked_rows.append(
            {
                "window": window_id,
                "train_start": train_prices.index[0],
                "train_end": train_prices.index[-1],
                "test_start": test_prices.index[0],
                "test_end": test_prices.index[-1],
                "picked": best.name,
                "cagr": float(test_metrics["cagr"]),
                "sharpe": float(test_metrics["sharpe"]),
                "mdd": float(test_metrics["mdd"]),
            }
        )
        oos_returns.append(test_metrics["returns"])  # type: ignore[arg-type]
        oos_weights.append(test_weights)

    stitched = pd.concat(oos_returns).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    stitched_weights = pd.concat(oos_weights).sort_index()
    stitched_weights = stitched_weights[~stitched_weights.index.duplicated(keep="last")]
    stitched_weights = stitched_weights.reindex(stitched.index).ffill().fillna(0.0)
    stitched_equity = (1.0 + stitched).cumprod()
    years = len(stitched) / 252
    rf_daily = (1.0 + rf) ** (1 / 252) - 1.0
    excess = stitched - rf_daily
    vol = stitched.std() * np.sqrt(252)
    mdd = (stitched_equity / stitched_equity.cummax() - 1.0).min()
    turnover = (stitched_weights.diff().abs().sum(axis=1) / 2.0).sum() / years

    return {
        "windows": picked_rows,
        "returns": stitched,
        "metrics": {
            "label": "WFO Stitched OOS",
            "weights": stitched_weights,
            "returns": stitched,
            "equity": stitched_equity,
            "cagr": stitched_equity.iloc[-1] ** (1 / years) - 1.0,
            "vol": vol,
            "sharpe": excess.mean() * 252 / vol if vol > 0 else np.nan,
            "mdd": mdd,
            "calmar": (stitched_equity.iloc[-1] ** (1 / years) - 1.0) / abs(mdd) if mdd < 0 else np.nan,
            "turnover": turnover,
            "avg_cash": stitched_weights["CASH"].mean(),
        },
    }


def benchmark_oos_metrics(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rf: float,
    tx_cost: float,
    cost_model: Optional[IndianDeliveryCostModel] = None,
    base_value: float = 1_000_000.0,
) -> List[Dict[str, object]]:
    subset = prices.loc[dates]
    names = ["EqWt Risky", "Nifty B&H", "Smallcap B&H", "60/40 Nifty/Cash"]
    return [
        performance_metrics(
            subset,
            benchmark_weights(subset, name),
            name,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )
        for name in names
    ]


def print_universe_summary(prices: pd.DataFrame) -> None:
    years = len(prices) / 252
    print("\nUniverse (cleaned INR series):")
    for asset in ALL:
        cagr = (prices[asset].iloc[-1] / prices[asset].iloc[0]) ** (1 / years) - 1.0
        max_daily = prices[asset].pct_change(fill_method=None).abs().max()
        print(f"  {asset:<10} CAGR {cagr:>7.1%} | max daily move {max_daily:>6.1%}")


def print_tradable_proxy_notes(prices: pd.DataFrame) -> None:
    starts = {asset: prices[asset].dropna().index[0].date() for asset in ALL}
    earliest_common = max(starts.values())
    print("\nTradable proxy mode notes:")
    print("  This mode replaces indices/synthetic sleeves with tradable NSE products.")
    print(f"  Earliest common start across all proxies: {earliest_common}")
    for asset in ALL:
        print(f"  {asset:<10} first tradable proxy bar {starts[asset]}")


def print_benchmark_proxy_notes() -> None:
    print("\nBenchmark proxy mode notes:")
    print("  This mode is for clear, consistent long-history benchmarking.")
    print("  NIFTY / MIDCAP / SMALLCAP use their corresponding indices.")
    print("  GOLD uses the India-listed GOLDBEES series.")
    print("  US uses the India-listed MON100 series.")
    if INDIA_SILVER_OVERRIDE_PATH.exists():
        print(f"  SILVER uses the India-only override file at {INDIA_SILVER_OVERRIDE_PATH}.")
    else:
        print("  SILVER falls back to global silver in INR unless you add config/data/silver_india_daily.csv.")
    print("  CASH is a synthetic benchmark sleeve; live execution maps it to LIQUIDBEES.")
    print("  Use this mode for benchmarking and research, not for live order generation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-turnover multi-asset tactical allocation backtest.")
    parser.add_argument("--start", default=DEFAULT_BACKTEST_START, help="Backtest start date.")
    parser.add_argument("--end", default=DEFAULT_BACKTEST_END, help="Backtest end date.")
    parser.add_argument("--rf", type=float, default=DEFAULT_RF, help="Annual risk-free rate.")
    parser.add_argument("--tx-bps", type=float, default=30.0, help="Transaction cost in basis points per trade.")
    parser.add_argument(
        "--cost-model",
        choices=["flat", "india_delivery"],
        default="flat",
        help="Use flat half-turnover bps or the official-style India delivery ETF/equity estimator.",
    )
    parser.add_argument("--base-value", type=float, default=1_000_000.0, help="Portfolio value used for fixed-cost backtest estimates.")
    parser.add_argument("--refresh-cache", action="store_true", help="Redownload price history instead of using cache.")
    parser.add_argument("--data-source", choices=["yfinance", "groww"], default="yfinance", help="Price source.")
    parser.add_argument(
        "--universe-mode",
        choices=["research", "benchmark", "tradable"],
        default="research",
        help="Research sleeves, benchmark backfilled sleeves, or pure tradable ETF proxy sleeves.",
    )
    parser.add_argument("--groww-smoke-test", action="store_true", help="Run Groww auth/profile/quote/candle smoke test before backtesting.")
    parser.add_argument("--groww-universe-file", help="Optional JSON file overriding default Groww tradable symbols.")
    parser.add_argument("--llm-override-file", help="Optional JSON file with per-date risk_off_override values.")
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000
    cost_model = resolve_cost_model(args.cost_model)

    recommended = StrategyConfig(
        name="weekly_core85_tilt15",
        execution_frequency="WEEKLY",
        core_weight=0.85,
        tilt_weight=0.15,
        top_n=2,
        trade_band=0.08,
        trade_step=0.75,
        crash_floor=0.70,
    )

    candidates = [
        recommended,
        StrategyConfig(
            name="weekly_core80_tilt20",
            execution_frequency="WEEKLY",
            core_weight=0.80,
            tilt_weight=0.20,
            top_n=2,
            trade_band=0.08,
            trade_step=0.75,
            crash_floor=0.65,
        ),
        StrategyConfig(
            name="monthly_core70_tilt30",
            execution_frequency="MONTHLY",
            core_weight=0.70,
            tilt_weight=0.30,
            top_n=2,
            trade_band=0.06,
            trade_step=1.00,
            crash_floor=0.65,
        ),
        StrategyConfig(
            name="monthly_core85_tilt15",
            execution_frequency="MONTHLY",
            core_weight=0.85,
            tilt_weight=0.15,
            top_n=2,
            trade_band=0.05,
            trade_step=1.00,
            crash_floor=0.70,
        ),
    ]

    print("=" * 88)
    print("ALPHA ENGINE v9 — LOW-TURNOVER CORE + TACTICAL TILT")
    print("=" * 88)
    print(f"Backtest window: {args.start} -> {args.end}")
    print(f"Transaction cost: {tx_cost:.2%} per trade")
    print(f"Cost model: {args.cost_model}")
    print(f"Risk-free rate: {args.rf:.2%}")
    print(f"Data source: {args.data_source}")
    print(f"Universe mode: {args.universe_mode}")

    groww_session: Optional[GrowwSession] = None
    if args.data_source == "groww" or args.groww_smoke_test:
        groww_session = GrowwSession.from_env(
            instrument_map=load_groww_universe(args.groww_universe_file),
            cache_dir=CACHE_DIR,
        )
        if args.groww_smoke_test:
            smoke = groww_session.smoke_test()
            print("\nGroww smoke test:")
            print(f"  Profile keys: {', '.join(smoke['profile_keys'])}")
            print(f"  Holdings count: {smoke['holdings_count']}")
            print(f"  Positions count: {smoke['positions_count']}")
            print(f"  Quote keys sampled: {', '.join(smoke['quote_keys'][:6])}")
            print(f"  LTP snapshot: {smoke['ltp']}")
            print(f"  Historical candles: {smoke['candles_count']} rows")

    if args.data_source == "groww":
        prices = GrowwSource(groww_session).fetch(args.start, refresh=args.refresh_cache)  # type: ignore[arg-type]
        prices = prices.loc[pd.Timestamp(args.start) : pd.Timestamp(args.end)]
        if pd.Timestamp(args.start) < GROWW_MIN_HISTORY_START:
            print(
                f"\nGroww note: official historical support is documented from {GROWW_MIN_HISTORY_START.date()} onward, "
                "so earlier backtests still need a non-Groww data vendor."
            )
    else:
        prices = YahooFinanceSource(universe_mode=args.universe_mode).fetch(args.start, end=args.end, refresh=args.refresh_cache)

    print(f"\nPrice rows: {len(prices)} | {prices.index[0].date()} -> {prices.index[-1].date()}")
    print_universe_summary(prices)
    if args.data_source == "yfinance" and args.universe_mode == "tradable":
        print_tradable_proxy_notes(prices)
    if args.data_source == "yfinance" and args.universe_mode == "benchmark":
        print_benchmark_proxy_notes()

    overlay = load_llm_overlay(args.llm_override_file, prices.index)
    if overlay is not None:
        active_days = int((overlay.risk_off > 0).sum())
        print(f"\nLoaded LLM override file: {args.llm_override_file} | active override days: {active_days}")

    full_sample_results: List[Dict[str, object]] = []
    for config in candidates:
        weights = run_strategy(prices, config, overlay=overlay)
        full_sample_results.append(
            performance_metrics(
                prices,
                weights,
                config.name,
                rf=args.rf,
                tx_cost=tx_cost,
                cost_model=cost_model,
                base_value=args.base_value,
            )
        )

    for name in ["EqWt Risky", "Nifty B&H", "Smallcap B&H", "60/40 Nifty/Cash"]:
        full_sample_results.append(
            performance_metrics(
                prices,
                benchmark_weights(prices, name),
                name,
                rf=args.rf,
                tx_cost=tx_cost,
                cost_model=cost_model,
                base_value=args.base_value,
            )
        )

    ordered_full_sample = sorted(
        full_sample_results,
        key=lambda row: (float(row["sharpe"]), float(row["calmar"])),
        reverse=True,
    )
    print_results_table("FULL SAMPLE", ordered_full_sample)

    wfo = walk_forward(
        prices,
        candidates,
        rf=args.rf,
        tx_cost=tx_cost,
        overlay=overlay,
        cost_model=cost_model,
        base_value=args.base_value,
    )
    oos_metrics = wfo["metrics"]  # type: ignore[assignment]
    oos_dates = oos_metrics["returns"].index  # type: ignore[index]
    oos_benchmarks = (
        benchmark_oos_metrics(
            prices,
            oos_dates,
            rf=args.rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=args.base_value,
        )
        if len(oos_dates) > 0
        else []
    )

    if oos_benchmarks:
        print_results_table(
            "WALK-FORWARD OOS (2Y TRAIN / 6M TEST, STITCHED TEST RETURNS)",
            [oos_metrics, *oos_benchmarks],  # type: ignore[list-item]
        )

        print("\nWindow picks:")
        for row in wfo["windows"]:  # type: ignore[index]
            print(
                f"  W{row['window']:>2} | "
                f"{pd.Timestamp(row['test_start']).strftime('%Y-%m')} -> {pd.Timestamp(row['test_end']).strftime('%Y-%m')} | "
                f"picked {row['picked']:<20} | "
                f"CAGR {row['cagr']:.1%} | Sharpe {row['sharpe']:.2f} | MaxDD {row['mdd']:.1%}"
            )
    else:
        print("\nWalk-forward skipped: not enough history for a 2-year train / 6-month test window.")

    eqwt_full = next(row for row in full_sample_results if row["label"] == "EqWt Risky")
    rec_full = next(row for row in full_sample_results if row["label"] == recommended.name)
    oos_eqwt = next((row for row in oos_benchmarks if row["label"] == "EqWt Risky"), None)

    print("\nWhat changed from v8:")
    print("  1. Daily data is still used, but daily trading is gone. Signals update daily; execution does not.")
    print("  2. The universe remains the alpha engine. The model now keeps a broad core instead of rotating everything.")
    print("  3. Risk control is explicit and fast: breadth + crash state replaced laggy Markov refits.")
    print("  4. Position sizing is robust inverse-vol, not noisy Kelly on unstable daily expected returns.")

    print("\nHonest read:")
    print(
        f"  Recommended full-sample result: {recommended.name} "
        f"CAGR {rec_full['cagr']:.1%}, Sharpe {rec_full['sharpe']:.2f}, "
        f"MaxDD {rec_full['mdd']:.1%}, turnover {rec_full['turnover']:.0%}."
    )
    print(
        f"  Equal-weight risky buy-and-hold is still stronger on pure return in this dataset: "
        f"CAGR {eqwt_full['cagr']:.1%}, Sharpe {eqwt_full['sharpe']:.2f}, MaxDD {eqwt_full['mdd']:.1%}."
    )
    if oos_eqwt is not None:
        print(
            f"  On stitched walk-forward OOS, the engine posts CAGR {oos_metrics['cagr']:.1%} "
            f"vs EqWt {oos_eqwt['cagr']:.1%} and Nifty "
            f"{next(row for row in oos_benchmarks if row['label'] == 'Nifty B&H')['cagr']:.1%}."
        )
    else:
        print("  Walk-forward OOS comparison was skipped because the supplied history was too short.")
    print("  Conclusion: this is a much better trading system than v8, but I cannot honestly claim it beats equal-weight risky buy-and-hold after costs.")

    RESULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "full_sample": {
            row["label"]: {
                "cagr": row["cagr"],
                "vol": row["vol"],
                "sharpe": row["sharpe"],
                "mdd": row["mdd"],
                "calmar": row["calmar"],
                "turnover": row["turnover"],
                "avg_cash": row["avg_cash"],
            }
            for row in full_sample_results
        },
        "wfo": {
            "stitched": {
                "cagr": oos_metrics["cagr"],
                "vol": oos_metrics["vol"],
                "sharpe": oos_metrics["sharpe"],
                "mdd": oos_metrics["mdd"],
                "calmar": oos_metrics["calmar"],
            },
            "windows": [
                {
                    "window": row["window"],
                    "picked": row["picked"],
                    "test_start": str(pd.Timestamp(row["test_start"]).date()),
                    "test_end": str(pd.Timestamp(row["test_end"]).date()),
                    "cagr": row["cagr"],
                    "sharpe": row["sharpe"],
                    "mdd": row["mdd"],
                }
                for row in wfo["windows"]  # type: ignore[index]
            ],
        },
        "meta": {
            "data_source": args.data_source,
            "universe_mode": args.universe_mode,
            "cost_model": args.cost_model,
            "tx_cost": tx_cost,
            "base_value": args.base_value,
            "price_start": prices.index[0].strftime("%Y-%m-%d"),
            "price_end": prices.index[-1].strftime("%Y-%m-%d"),
        },
    }
    with RESULTS_CACHE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)


if __name__ == "__main__":
    main()
