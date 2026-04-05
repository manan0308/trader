#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from market_data.market_store import LOCAL_SILVER_OVERRIDE_PATH


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "market" / "raw" / "india_silver"
ARCHIVE_PATH = RAW_DIR / "india_silver_normalized.pkl"
MANIFEST_PATH = RAW_DIR / "manifest.json"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SILVER_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError(f"{path} does not contain a CSV file.")
            with archive.open(csv_names[0]) as handle:
                return pd.read_csv(handle)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def pick_column(columns: Dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def normalized_columns(frame: pd.DataFrame) -> Dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in frame.columns}


def text_mask(frame: pd.DataFrame, column_names: Iterable[str], needles: Iterable[str]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    lowered = [needle.lower() for needle in needles]
    for name in column_names:
        if name not in frame.columns:
            continue
        values = frame[name].astype(str).str.lower()
        mask |= values.apply(lambda value: any(needle in value for needle in lowered))
    return mask


def normalize_mcx(frame: pd.DataFrame) -> pd.DataFrame:
    cols = normalized_columns(frame)
    date_col = pick_column(cols, ["date", "tradedate", "timestamp", "trade_date"])
    close_col = pick_column(cols, ["settlepr", "settle", "close", "settlementprice", "price"])
    volume_col = pick_column(cols, ["volume", "vol", "tradevalueqty"])
    oi_col = pick_column(cols, ["openinterest", "oi"])
    expiry_col = pick_column(cols, ["expirydate", "expiry", "contractexpiry"])
    contract_col = pick_column(cols, ["symbol", "contract", "commodity", "instrumentname", "underlying"])
    if date_col is None or close_col is None:
        raise ValueError("MCX input must contain date and settle/close columns.")

    working = frame.copy()
    filter_columns = [col for col in [contract_col, pick_column(cols, ["instrumentname", "underlying", "commodity"])] if col]
    if filter_columns:
        mask = text_mask(working, filter_columns, ["silver"])
        working = working.loc[mask].copy()
    if working.empty:
        raise ValueError("No SILVER rows found in MCX input.")

    working["date"] = pd.to_datetime(working[date_col]).dt.normalize()
    working["close"] = pd.to_numeric(working[close_col], errors="coerce")
    working["volume"] = pd.to_numeric(working[volume_col], errors="coerce") if volume_col else 0.0
    working["open_interest"] = pd.to_numeric(working[oi_col], errors="coerce") if oi_col else 0.0
    working["contract"] = working[contract_col].astype(str) if contract_col else "SILVER"
    working["expiry"] = pd.to_datetime(working[expiry_col], errors="coerce") if expiry_col else pd.NaT
    working = working.dropna(subset=["date", "close"])
    working["source"] = "mcx"
    working["source_file"] = ""
    return working[["date", "close", "volume", "open_interest", "contract", "expiry", "source", "source_file"]]


def normalize_ibkr(frame: pd.DataFrame) -> pd.DataFrame:
    cols = normalized_columns(frame)
    date_col = pick_column(cols, ["date", "datetime", "time"])
    close_col = pick_column(cols, ["close", "settle", "last"])
    volume_col = pick_column(cols, ["volume"])
    contract_col = pick_column(cols, ["localsymbol", "symbol", "description", "conid"])
    expiry_col = pick_column(cols, ["expiry", "lasttradedate", "lasttradingday"])
    if date_col is None or close_col is None:
        raise ValueError("IBKR input must contain date and close columns.")

    working = frame.copy()
    filter_columns = [col for col in [contract_col, pick_column(cols, ["description"])] if col]
    if filter_columns:
        mask = text_mask(working, filter_columns, ["silver", " si", "si "])
        if mask.any():
            working = working.loc[mask].copy()
    working["date"] = pd.to_datetime(working[date_col]).dt.normalize()
    working["close"] = pd.to_numeric(working[close_col], errors="coerce")
    working["volume"] = pd.to_numeric(working[volume_col], errors="coerce") if volume_col else 0.0
    working["open_interest"] = 0.0
    working["contract"] = working[contract_col].astype(str) if contract_col else "IBKR_SILVER"
    working["expiry"] = pd.to_datetime(working[expiry_col], errors="coerce") if expiry_col else pd.NaT
    working = working.dropna(subset=["date", "close"])
    working["source"] = "ibkr"
    working["source_file"] = ""
    return working[["date", "close", "volume", "open_interest", "contract", "expiry", "source", "source_file"]]


def normalize_generic(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    cols = normalized_columns(frame)
    date_col = pick_column(cols, ["date", "datetime", "time"])
    close_col = pick_column(cols, ["close", "settle", "price", "value", "last"])
    volume_col = pick_column(cols, ["volume"])
    contract_col = pick_column(cols, ["contract", "symbol", "name", "description"])
    if date_col is None or close_col is None:
        raise ValueError("Generic input must contain date and close columns.")
    working = frame.copy()
    working["date"] = pd.to_datetime(working[date_col]).dt.normalize()
    working["close"] = pd.to_numeric(working[close_col], errors="coerce")
    working["volume"] = pd.to_numeric(working[volume_col], errors="coerce") if volume_col else 0.0
    working["open_interest"] = 0.0
    working["contract"] = working[contract_col].astype(str) if contract_col else source_name.upper()
    working["expiry"] = pd.NaT
    working = working.dropna(subset=["date", "close"])
    working["source"] = source_name
    working["source_file"] = ""
    return working[["date", "close", "volume", "open_interest", "contract", "expiry", "source", "source_file"]]


def choose_daily_close(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["expiry_sort"] = pd.to_datetime(working["expiry"], errors="coerce").fillna(pd.Timestamp.max)
    working["volume"] = pd.to_numeric(working["volume"], errors="coerce").fillna(0.0)
    working["open_interest"] = pd.to_numeric(working["open_interest"], errors="coerce").fillna(0.0)
    ordered = working.sort_values(
        by=["date", "volume", "open_interest", "expiry_sort"],
        ascending=[True, False, False, True],
    )
    chosen = ordered.groupby("date", as_index=False).first()
    chosen["roll_id"] = (chosen["contract"] != chosen["contract"].shift(1)).cumsum().astype(int)
    return chosen


def load_existing_archive() -> pd.DataFrame:
    if not ARCHIVE_PATH.exists():
        return pd.DataFrame(columns=["date", "close", "volume", "open_interest", "contract", "expiry", "source", "source_file"])
    frame = pd.read_pickle(ARCHIVE_PATH)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def save_outputs(raw_rows: pd.DataFrame, canonical: pd.DataFrame, output_path: Path) -> None:
    raw_rows.to_pickle(ARCHIVE_PATH)
    manifest = {
        "rows": int(len(raw_rows)),
        "canonical_rows": int(len(canonical)),
        "start": canonical["date"].min().strftime("%Y-%m-%d") if len(canonical) else None,
        "end": canonical["date"].max().strftime("%Y-%m-%d") if len(canonical) else None,
        "output_path": str(output_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = canonical.rename(columns={"date": "Date", "close": "Close", "source": "Source", "contract": "Contract", "volume": "Volume", "open_interest": "OpenInterest", "roll_id": "RollId"})
    output.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize MCX or IBKR India silver history into the local data lake.")
    parser.add_argument("--input", action="append", required=True, help="One or more local MCX/IBKR/CSV files.")
    parser.add_argument("--source", choices=["mcx", "ibkr", "generic", "auto"], default="auto")
    parser.add_argument("--output", default=str(LOCAL_SILVER_OVERRIDE_PATH))
    parser.add_argument("--replace", action="store_true", help="Replace the archive instead of merging into it.")
    args = parser.parse_args()

    ensure_dirs()
    all_rows: List[pd.DataFrame] = []
    for raw_path in args.input:
        path = Path(raw_path).expanduser().resolve()
        frame = load_input(path)
        source = args.source
        if source == "auto":
            lower_name = path.name.lower()
            cols = normalized_columns(frame)
            if "settlepr" in cols or "tradedate" in cols or "openinterest" in cols:
                source = "mcx"
            elif "localsymbol" in cols or "barcount" in cols:
                source = "ibkr"
            elif "mcx" in lower_name:
                source = "mcx"
            elif "ibkr" in lower_name or "interactive" in lower_name:
                source = "ibkr"
            else:
                source = "generic"

        if source == "mcx":
            normalized = normalize_mcx(frame)
        elif source == "ibkr":
            normalized = normalize_ibkr(frame)
        else:
            normalized = normalize_generic(frame, source_name=source)
        normalized["source_file"] = str(path)
        all_rows.append(normalized)

    combined = pd.concat(all_rows, ignore_index=True)
    existing = pd.DataFrame() if args.replace else load_existing_archive()
    merged = pd.concat([existing, combined], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.drop_duplicates(subset=["date", "close", "contract", "source"], keep="last")
    merged = merged.sort_values(["date", "source", "contract"]).reset_index(drop=True)

    canonical = choose_daily_close(merged)
    output_path = Path(args.output).expanduser().resolve()
    save_outputs(merged, canonical, output_path)

    print(json.dumps(
        {
            "archive_rows": int(len(merged)),
            "canonical_rows": int(len(canonical)),
            "start": canonical["date"].min().strftime("%Y-%m-%d") if len(canonical) else None,
            "end": canonical["date"].max().strftime("%Y-%m-%d") if len(canonical) else None,
            "output": str(output_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
