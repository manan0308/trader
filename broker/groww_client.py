from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from runtime.env_loader import load_runtime_env

try:
    from growwapi import GrowwAPI
except ImportError:  # pragma: no cover - optional dependency at runtime
    GrowwAPI = None


GROWW_MIN_HISTORY_START = pd.Timestamp("2020-01-01")
GROWW_MAX_DAY_CHUNK = 180

GROWW_API_KEY_ENV = "GROWW_API_KEY"
GROWW_API_SECRET_ENV = "GROWW_API_SECRET"
GROWW_ACCESS_TOKEN_ENV = "GROWW_ACCESS_TOKEN"


@dataclass(frozen=True)
class GrowwInstrument:
    asset: str
    exchange: str
    segment: str
    trading_symbol: str
    groww_symbol: str


DEFAULT_GROWW_UNIVERSE: Dict[str, GrowwInstrument] = {
    "NIFTY": GrowwInstrument("NIFTY", "NSE", "CASH", "NIFTYBEES", "NSE-NIFTYBEES"),
    "MIDCAP": GrowwInstrument("MIDCAP", "NSE", "CASH", "MID150BEES", "NSE-MID150BEES"),
    # This is a proxy ETF symbol, not the original index.
    "SMALLCAP": GrowwInstrument("SMALLCAP", "NSE", "CASH", "SMALLCAP", "NSE-SMALLCAP"),
    "GOLD": GrowwInstrument("GOLD", "NSE", "CASH", "GOLDBEES", "NSE-GOLDBEES"),
    "SILVER": GrowwInstrument("SILVER", "NSE", "CASH", "SILVERBEES", "NSE-SILVERBEES"),
    "US": GrowwInstrument("US", "NSE", "CASH", "MON100", "NSE-MON100"),
    "CASH": GrowwInstrument("CASH", "NSE", "CASH", "LIQUIDBEES", "NSE-LIQUIDBEES"),
}

PRODUCTION_GROWW_UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "config" / "groww_universe.production.json"


def clean_price_series(series: pd.Series) -> pd.Series:
    clean = pd.Series(series).copy().astype(float)
    if getattr(clean.index, "tz", None) is not None:
        clean.index = clean.index.tz_localize(None)
    clean = clean.sort_index()

    rolling_median = clean.rolling(20, min_periods=5).median()
    level_bad = ((clean / rolling_median) > 4.0) | ((clean / rolling_median) < 0.25)
    clean[level_bad] = np.nan

    for _ in range(2):
        jump_bad = clean.pct_change(fill_method=None).abs() > 0.50
        clean[jump_bad] = np.nan

    return clean.ffill(limit=5)


def load_groww_universe(path: Optional[str] = None, *, prefer_production: bool = False) -> Dict[str, GrowwInstrument]:
    if not path and prefer_production and PRODUCTION_GROWW_UNIVERSE_PATH.exists():
        path = str(PRODUCTION_GROWW_UNIVERSE_PATH)

    if not path:
        return DEFAULT_GROWW_UNIVERSE

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    universe: Dict[str, GrowwInstrument] = {}
    for asset, data in payload.items():
        exchange = data["exchange"]
        trading_symbol = data["trading_symbol"]
        universe[asset] = GrowwInstrument(
            asset=asset,
            exchange=exchange,
            segment=data.get("segment", "CASH"),
            trading_symbol=trading_symbol,
            groww_symbol=data.get("groww_symbol", f"{exchange}-{trading_symbol}"),
        )
    return universe


def load_production_groww_universe(path: Optional[str] = None) -> Dict[str, GrowwInstrument]:
    return load_groww_universe(path, prefer_production=True)


def chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int = GROWW_MAX_DAY_CHUNK) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


class GrowwSession:
    def __init__(
        self,
        access_token: str,
        instrument_map: Optional[Dict[str, GrowwInstrument]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if GrowwAPI is None:
            raise RuntimeError("growwapi is not installed. Run `pip install growwapi` first.")
        self.access_token = access_token
        self.client = GrowwAPI(access_token)
        self.instrument_map = instrument_map or DEFAULT_GROWW_UNIVERSE
        self.cache_dir = cache_dir or Path.cwd() / "cache"

    @classmethod
    def from_env(
        cls,
        instrument_map: Optional[Dict[str, GrowwInstrument]] = None,
        cache_dir: Optional[Path] = None,
    ) -> "GrowwSession":
        load_runtime_env(override=True)
        access_token = os.getenv(GROWW_ACCESS_TOKEN_ENV)
        if not access_token:
            api_key = os.getenv(GROWW_API_KEY_ENV)
            api_secret = os.getenv(GROWW_API_SECRET_ENV)
            if not api_key or not api_secret:
                raise RuntimeError(
                    "Missing Groww credentials. Set either GROWW_ACCESS_TOKEN or "
                    "both GROWW_API_KEY and GROWW_API_SECRET."
                )
            access_token = GrowwAPI.get_access_token(api_key=api_key, secret=api_secret)
        return cls(access_token=access_token, instrument_map=instrument_map, cache_dir=cache_dir)

    def smoke_test(self) -> Dict[str, object]:
        profile = self.client.get_user_profile()
        holdings = self.client.get_holdings_for_user()
        positions = self.client.get_positions_for_user(segment=self.client.SEGMENT_CASH)
        quote = self.client.get_quote(
            trading_symbol=self.instrument_map["NIFTY"].trading_symbol,
            exchange=self.client.EXCHANGE_NSE,
            segment=self.client.SEGMENT_CASH,
        )
        ltp = self.client.get_ltp(
            exchange_trading_symbols=(
                "NSE_NIFTYBEES",
                "NSE_MON100",
            ),
            segment=self.client.SEGMENT_CASH,
        )
        candles = self.client.get_historical_candles(
            exchange=self.client.EXCHANGE_NSE,
            segment=self.client.SEGMENT_CASH,
            groww_symbol=self.instrument_map["NIFTY"].groww_symbol,
            start_time="2025-01-01 09:15:00",
            end_time="2025-02-01 15:30:00",
            candle_interval=self.client.CANDLE_INTERVAL_DAY,
        )

        holdings_count = len(holdings.get("holdings", [])) if isinstance(holdings, dict) else 0
        positions_count = len(positions.get("positions", [])) if isinstance(positions, dict) else 0
        candles_count = len(candles.get("candles", [])) if isinstance(candles, dict) else 0

        return {
            "profile_keys": sorted(profile.keys()) if isinstance(profile, dict) else [],
            "holdings_count": holdings_count,
            "positions_count": positions_count,
            "quote_keys": sorted(quote.keys()) if isinstance(quote, dict) else [],
            "ltp": ltp,
            "candles_count": candles_count,
            "first_candle": candles.get("candles", [None])[0] if isinstance(candles, dict) else None,
        }

    def get_all_instruments(self) -> pd.DataFrame:
        return self.client.get_all_instruments()

    def fetch_universe_prices(self, start: str, refresh: bool = False) -> pd.DataFrame:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        effective_start = max(pd.Timestamp(start), GROWW_MIN_HISTORY_START)
        end = pd.Timestamp.utcnow().tz_localize(None).normalize()

        if effective_start > end:
            raise RuntimeError("Groww start date is later than today.")

        # Include the end date in the cache key. Previously the cache file was
        # only keyed by ``start`` so re-running on subsequent days would keep
        # reading the first-day cache forever, silently hiding every new
        # candle from downstream consumers.
        cache_name = (
            "groww_prices_"
            f"{pd.Timestamp(start).strftime('%Y%m%d')}_to_{end.strftime('%Y%m%d')}.csv"
        )
        cache_path = self.cache_dir / cache_name

        if cache_path.exists() and not refresh:
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return cached.sort_index()

        prices = pd.DataFrame()
        for asset, instrument in self.instrument_map.items():
            prices[asset] = self.fetch_asset_history(instrument, effective_start, end)

        prices = prices.ffill(limit=5).dropna()
        prices.to_csv(cache_path, index=True)
        return prices

    def fetch_asset_history(
        self,
        instrument: GrowwInstrument,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        rows: list[tuple[pd.Timestamp, float]] = []
        for chunk_start, chunk_end in chunk_ranges(start, end):
            response = self._get_historical_chunk(instrument, chunk_start, chunk_end)
            for candle in response.get("candles", []):
                timestamp = pd.Timestamp(candle[0])
                close = float(candle[4])
                rows.append((timestamp, close))
            time.sleep(0.25)

        if not rows:
            raise RuntimeError(f"No Groww candles returned for {instrument.asset} ({instrument.trading_symbol}).")

        series = pd.Series(
            data=[close for _, close in rows],
            index=[timestamp for timestamp, _ in rows],
            name=instrument.asset,
        )
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return clean_price_series(series)

    def quote(self, asset: str) -> Dict[str, object]:
        instrument = self.instrument_map[asset]
        return self.client.get_quote(
            trading_symbol=instrument.trading_symbol,
            exchange=getattr(self.client, f"EXCHANGE_{instrument.exchange}"),
            segment=getattr(self.client, f"SEGMENT_{instrument.segment}"),
        )

    def multi_ltp(self, assets: Iterable[str]) -> Dict[str, float]:
        symbols = tuple(f"{self.instrument_map[asset].exchange}_{self.instrument_map[asset].trading_symbol}" for asset in assets)
        return self.client.get_ltp(
            exchange_trading_symbols=symbols,
            segment=self.client.SEGMENT_CASH,
        )

    def _get_historical_chunk(
        self,
        instrument: GrowwInstrument,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict:
        last_error: Optional[Exception] = None
        for attempt in range(5):
            try:
                return self.client.get_historical_candles(
                    exchange=getattr(self.client, f"EXCHANGE_{instrument.exchange}"),
                    segment=getattr(self.client, f"SEGMENT_{instrument.segment}"),
                    groww_symbol=instrument.groww_symbol,
                    start_time=start.strftime("%Y-%m-%d 09:15:00"),
                    end_time=end.strftime("%Y-%m-%d 15:30:00"),
                    candle_interval=self.client.CANDLE_INTERVAL_DAY,
                )
            except Exception as exc:
                last_error = exc
                if "Rate limit" not in str(exc) and "rate limit" not in str(exc):
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(
            f"Historical fetch failed for {instrument.asset} "
            f"({instrument.trading_symbol}) after retries: {last_error}"
        )
