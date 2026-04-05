#!/usr/bin/env python3
from __future__ import annotations

import argparse

from market_data.market_store import build_and_save_dataset, dataset_names


DEFAULT_STARTS = {
    "india_research": "2012-04-01",
    "india_benchmark": "2012-04-01",
    "india_tradable": "2024-01-01",
    "india_macro": "2012-04-01",
    "us_analog_shy": "2006-01-01",
    "us_analog_ief": "2006-01-01",
    "us_analog_bil": "2006-01-01",
}

DEFAULT_ENDS = {
    "india_research": None,
    "india_benchmark": None,
    "india_tradable": None,
    "india_macro": None,
    "us_analog_shy": None,
    "us_analog_ief": None,
    "us_analog_bil": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and save offline market datasets for research and backtests.")
    parser.add_argument("--dataset", action="append", choices=dataset_names(), help="Specific dataset(s) to build.")
    parser.add_argument("--all", action="store_true", help="Build all supported datasets.")
    parser.add_argument("--refresh", action="store_true", help="Redownload raw bars even if cached.")
    args = parser.parse_args()

    selected = args.dataset or []
    if args.all or not selected:
        selected = dataset_names()

    print("Offline dataset build")
    for name in selected:
        start = DEFAULT_STARTS[name]
        end = DEFAULT_ENDS[name]
        path = build_and_save_dataset(name, start=start, end=end, refresh=args.refresh)
        print(f"  {name:<16} -> {path}")


if __name__ == "__main__":
    main()
