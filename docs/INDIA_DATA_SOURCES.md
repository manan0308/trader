# India Data Sources

## Silver benchmark history

The repo currently supports a local India-only silver override file at:

- `config/data/silver_india_daily.csv`

This is the recommended path because public web sources for long-history Indian silver are inconsistent or bot-protected.

## Sources investigated

### Official / primary

- MCX historical and bhavcopy pages:
  - `https://www.mcxindia.com/market-data/historical-data`
  - `https://www.mcxindia.com/market-data/bhavcopy`
  - These are the cleanest primary-source candidates, but they returned access-denied responses in automation tests from this environment.

### Practical / engineering

- TradingView / `tvDatafeed`
  - Can be useful for MCX continuous futures history in some environments, but anonymous access was unreliable in tests here.

- Investing / MCX Silver Futures
  - `investpy` static metadata includes:
    - `MCX Silver` with `curr_id=49791`
    - `MCX Silver Micro` with `curr_id=49792`
    - `MCX Silver Mini` with `curr_id=49793`
  - Direct historical endpoint tests were blocked by Cloudflare from this environment.

- GitHub / community patterns
  - `Khushalsawant/Download-Bhavcopy-from-MCX_India`
  - `marketcalls/openalgo-python-library`
  - `joemccann/market-data-warehouse`
  - These are useful references for acquisition patterns, even if they are not the final production source.

### Broker / warehouse path

- IBKR historical bars
  - Useful as a broker-aligned supplement and for building a local data lake.
  - Stronger fit for a local warehouse or futures research workflow than for being the sole long-history primary source for India silver.

## Best next step

If you want a strict India-only silver sleeve for `2012-04-01` to `2026-03-31`, the most reliable path is:

1. export or acquire the domestic silver history from a trusted source,
2. normalize it with:

```bash
./.venv/bin/python -m trader_system.data.ingest_india_silver --input /path/to/mcx_or_ibkr_file.csv --source mcx
```

3. save the canonical output as `config/data/silver_india_daily.csv`,
4. rerun benchmark backtests.

The ingestor supports:

- `--source mcx`
- `--source ibkr`
- `--source generic`

and writes:

- the canonical close series used by the engine
- a raw normalized archive under `data/market/raw/india_silver/`

This keeps the strategy code stable while letting the data source improve independently.
