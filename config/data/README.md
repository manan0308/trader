# Local Data Overrides

Drop local benchmark-quality price histories here when a public API is weak or incomplete.

Current supported file:

- `silver_india_daily.csv`

Expected columns:

- `Date`
- one of `Close`, `Settle`, `Price`, `Value`, or `Last`

Example:

```csv
Date,Close
2012-04-02,56780
2012-04-03,57110
```

Notes:

- Dates should be daily trading dates in `YYYY-MM-DD` format.
- Values should be India-denominated silver prices for the chosen benchmark source.
- Once this file exists, `python -m trader_system.strategy.v9_engine --universe-mode benchmark` will use it automatically for the `SILVER` sleeve.
