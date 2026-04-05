from __future__ import annotations

import json
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE_DIR / "data" / "market"
RAW_DIR = DATA_ROOT / "raw" / "yfinance"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
LOCAL_SILVER_OVERRIDE_PATH = BASE_DIR / "config" / "data" / "silver_india_daily.csv"
PARQUET_AVAILABLE = find_spec("pyarrow") is not None or find_spec("fastparquet") is not None

INDIA_RESEARCH_TICKERS = {
    "NIFTY": "^NSEI",
    "MIDCAP": "^NSEMDCP50",
    "SMALLCAP": "^CRSMID",
    "GOLD": "GOLDBEES.NS",
    "US": "MON100.NS",
    "CASH": "LIQUIDBEES.NS",
    "SI": "SI=F",
    "FX": "INR=X",
}
INDIA_BENCHMARK_TICKERS = {
    "NIFTY": "^NSEI",
    "MIDCAP": "^NSEMDCP50",
    "SMALLCAP": "^CRSMID",
    "GOLD": "GOLDBEES.NS",
    "US": "MON100.NS",
    "SI": "SI=F",
    "FX": "INR=X",
}
INDIA_TRADABLE_TICKERS = {
    "NIFTY": "NIFTYBEES.NS",
    "MIDCAP": "MID150BEES.NS",
    "SMALLCAP": "SMALLCAP.NS",
    "GOLD": "GOLDBEES.NS",
    "SILVER": "SILVERBEES.NS",
    "US": "MON100.NS",
    "CASH": "LIQUIDBEES.NS",
}
INDIA_MACRO_TICKERS = {
    "INDIAVIX": "^INDIAVIX",
    "USDINR": "INR=X",
    "CRUDE": "CL=F",
    "US10Y": "^TNX",
}
US_ANALOG_UNIVERSES = {
    "us_analog_shy": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "SHY",
    },
    "us_analog_ief": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "IEF",
    },
    "us_analog_bil": {
        "NIFTY": "SPY",
        "MIDCAP": "MDY",
        "SMALLCAP": "IWM",
        "GOLD": "GLD",
        "SILVER": "SLV",
        "US": "QQQ",
        "CASH": "BIL",
    },
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_ticker(ticker: str) -> str:
    return (
        ticker.replace("^", "_idx_")
        .replace("=", "_eq_")
        .replace("/", "_slash_")
        .replace(".", "_dot_")
        .replace("-", "_dash_")
    )


def clean_price_series(series: pd.Series) -> pd.Series:
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


def load_local_price_override(path: Path = LOCAL_SILVER_OVERRIDE_PATH) -> pd.Series | None:
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
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    return clean_price_series(series.sort_index())


def _frame_candidates(stem: Path) -> list[Path]:
    return [stem.with_suffix(".parquet"), stem.with_suffix(".pkl")]


def _preferred_frame_path(stem: Path) -> Path:
    return stem.with_suffix(".parquet" if PARQUET_AVAILABLE else ".pkl")


def _load_frame(stem: Path) -> pd.DataFrame | None:
    for path in _frame_candidates(stem):
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_pickle(path)
    return None


def _save_frame(stem: Path, frame: pd.DataFrame) -> Path:
    path = _preferred_frame_path(stem)
    if path.suffix == ".parquet":
        frame.to_parquet(path)
    else:
        frame.to_pickle(path)
    return path


def raw_path_for_ticker(ticker: str) -> Path:
    return _preferred_frame_path(RAW_DIR / sanitize_ticker(ticker))


def processed_path(name: str) -> Path:
    return _preferred_frame_path(PROCESSED_DIR / name)


def load_raw_bars(ticker: str) -> pd.DataFrame | None:
    frame = _load_frame(RAW_DIR / sanitize_ticker(ticker))
    if frame is None:
        return None
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    return frame.sort_index()


def download_raw_bars(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            history = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True,
                repair=True,
            )
            if history.empty:
                raise RuntimeError(f"empty history for {ticker}")
            frame = history.copy()
            if getattr(frame.index, "tz", None) is not None:
                frame.index = frame.index.tz_localize(None)
            frame.index = pd.to_datetime(frame.index)
            return frame.sort_index()
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {ticker}: {last_error}")


def get_raw_bars(ticker: str, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if not refresh:
        cached = load_raw_bars(ticker)
        if cached is not None and len(cached) > 0:
            sliced = cached.loc[pd.Timestamp(start) :]
            if end:
                sliced = sliced.loc[: pd.Timestamp(end)]
            if len(sliced) > 0:
                return sliced
    frame = download_raw_bars(ticker, start=start, end=end)
    _save_frame(RAW_DIR / sanitize_ticker(ticker), frame)
    return frame


def _slice(df: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    out = df.sort_index().loc[pd.Timestamp(start) :]
    if end:
        out = out.loc[: pd.Timestamp(end)]
    return out


def _series_close(frame: pd.DataFrame) -> pd.Series:
    if "Close" not in frame.columns:
        raise ValueError("Close column missing from raw bars.")
    close = frame["Close"].copy()
    if "Repaired?" in frame.columns and {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        repaired = frame["Repaired?"].fillna(False).astype(bool)
        degenerate_bar = frame[["Open", "High", "Low", "Close"]].nunique(axis=1).le(1)
        jump_bar = close.pct_change(fill_method=None).abs().gt(0.08)
        # A repaired bar with a single repeated OHLC value and a large jump is usually
        # a vendor artifact rather than a real tradable close. Drop it and let the
        # local cleaner forward-fill briefly.
        close.loc[repaired & degenerate_bar & jump_bar] = np.nan
    return clean_price_series(close)


def _align_frames(series_map: Dict[str, pd.Series], columns: Iterable[str]) -> pd.DataFrame:
    frame = pd.DataFrame(series_map).ffill(limit=5).dropna()
    return frame[list(columns)]


def build_india_research_matrix(start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    raw = {name: get_raw_bars(ticker, start, end, refresh=refresh) for name, ticker in INDIA_RESEARCH_TICKERS.items()}
    prices = pd.DataFrame()
    for asset in ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "US", "CASH"]:
        prices[asset] = _series_close(raw[asset])
    local_silver = load_local_price_override()
    if local_silver is not None:
        prices["SILVER"] = local_silver
    else:
        si = _series_close(raw["SI"])
        fx = _series_close(raw["FX"])
        common = si.index.intersection(fx.index)
        prices["SILVER"] = clean_price_series(si.loc[common] * fx.loc[common])
    prices = prices.ffill(limit=5).dropna()
    return _slice(prices[["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]], start, end)


def build_india_benchmark_matrix(start: str, end: str | None = None, refresh: bool = False, rf: float = 0.065) -> pd.DataFrame:
    raw = {name: get_raw_bars(ticker, start, end, refresh=refresh) for name, ticker in INDIA_BENCHMARK_TICKERS.items()}
    prices = pd.DataFrame()
    for asset in ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "US"]:
        prices[asset] = _series_close(raw[asset])
    local_silver = load_local_price_override()
    if local_silver is not None:
        prices["SILVER"] = local_silver
    else:
        si = _series_close(raw["SI"])
        fx = _series_close(raw["FX"])
        common = si.index.intersection(fx.index)
        prices["SILVER"] = clean_price_series(si.loc[common] * fx.loc[common])
    prices = prices.ffill(limit=5).dropna()
    rf_daily = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    prices["CASH"] = 100.0 * np.power(1.0 + rf_daily, np.arange(len(prices), dtype=float))
    return _slice(prices[["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]], start, end)


def build_india_tradable_matrix(start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    raw = {name: get_raw_bars(ticker, start, end, refresh=refresh) for name, ticker in INDIA_TRADABLE_TICKERS.items()}
    prices = pd.DataFrame({asset: _series_close(raw[asset]) for asset in INDIA_TRADABLE_TICKERS})
    prices = prices.ffill(limit=5).dropna()
    return _slice(prices[["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]], start, end)


def build_macro_matrix(start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    raw = {name: get_raw_bars(ticker, start, end, refresh=refresh) for name, ticker in INDIA_MACRO_TICKERS.items()}
    panel = pd.DataFrame({name: _series_close(frame) for name, frame in raw.items()}).ffill(limit=5).dropna()
    return _slice(panel, start, end)


def build_us_matrix(name: str, start: str, end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    mapping = US_ANALOG_UNIVERSES[name]
    raw = {sleeve: get_raw_bars(ticker, start, end, refresh=refresh) for sleeve, ticker in mapping.items()}
    prices = pd.DataFrame({asset: _series_close(frame) for asset, frame in raw.items()}).ffill(limit=5).dropna()
    return _slice(prices[["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]], start, end)


BUILDERS = {
    "india_research": build_india_research_matrix,
    "india_benchmark": build_india_benchmark_matrix,
    "india_tradable": build_india_tradable_matrix,
    "india_macro": build_macro_matrix,
    "us_analog_shy": lambda start, end=None, refresh=False: build_us_matrix("us_analog_shy", start, end, refresh),
    "us_analog_ief": lambda start, end=None, refresh=False: build_us_matrix("us_analog_ief", start, end, refresh),
    "us_analog_bil": lambda start, end=None, refresh=False: build_us_matrix("us_analog_bil", start, end, refresh),
}


def load_processed_matrix(name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    frame = _load_frame(PROCESSED_DIR / name)
    if frame is None:
        return None
    frame.index = pd.to_datetime(frame.index)
    if start:
        frame = frame.loc[pd.Timestamp(start) :]
    if end:
        frame = frame.loc[: pd.Timestamp(end)]
    return frame.sort_index()


def save_processed_matrix(name: str, frame: pd.DataFrame) -> Path:
    ensure_dirs()
    return _save_frame(PROCESSED_DIR / name, frame.sort_index())


def build_and_save_dataset(name: str, start: str, end: str | None = None, refresh: bool = False) -> Path:
    if name not in BUILDERS:
        raise ValueError(f"unsupported dataset: {name}")
    frame = BUILDERS[name](start=start, end=end, refresh=refresh)
    path = save_processed_matrix(name, frame)
    update_manifest(name, frame)
    return path


def update_manifest(name: str, frame: pd.DataFrame) -> None:
    ensure_dirs()
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest[name] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "start": frame.index[0].strftime("%Y-%m-%d") if len(frame) else None,
        "end": frame.index[-1].strftime("%Y-%m-%d") if len(frame) else None,
        "path": str(processed_path(name)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dataset_names() -> list[str]:
    return list(BUILDERS.keys())
