# Offline Market Data Store

This folder is the local offline data lake for the repo.

It is meant to make backtests and research fast and repeatable without repeated API calls.

## Layout

- `raw/yfinance/`
  - raw per-ticker OHLCV cache files
- `processed/`
  - aligned close matrices used by research and backtests
- `manifest.json`
  - quick summary of what is currently available

## Build

From repo root:

```bash
./.venv/bin/python -m trader_system.data.download_offline_data --all
```

Use `--refresh` to refresh the raw bars before rebuilding:

```bash
./.venv/bin/python -m trader_system.data.download_offline_data --all --refresh
```

## Current processed datasets

- `india_research`
- `india_benchmark`
- `india_tradable`
- `india_macro`
- `us_analog_shy`
- `us_analog_ief`
- `us_analog_bil`

## Notes

- The strategy loaders now prefer these local processed datasets when available.
- In this environment the store falls back to `.pkl` files because `pyarrow` is not installed. If parquet support is added later, the same code can read either format.
- The India research and benchmark stores can extend beyond the research cutoff date. Backtests still stay honest because the strategy scripts slice to the explicit backtest window they were asked to use.
- The store is gitignored because the files can be large.
- For India silver, the canonical override file is still:
  - `config/data/silver_india_daily.csv`
- You can build that file from local MCX or IBKR exports using:

```bash
./.venv/bin/python -m trader_system.data.ingest_india_silver --input /path/to/file.csv --source mcx
```
