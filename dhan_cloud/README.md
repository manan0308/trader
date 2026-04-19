# Dhan Cloud Strategies

This folder contains thin strategy entrypoints meant for a Dhan Cloud style
Python runner.

They do three things:

1. fetch recent daily ETF candles directly from Dhan,
2. run the repo's strategy logic,
3. print a JSON payload with the latest target weights.

## Included strategies

- `v9_quant_strategy.py`
  - the current diversified baseline
- `v21_obv5_top2_daily_strategy.py`
  - the highest-CAGR challenger from the ETF price/volume research
  - this is intentionally aggressive and high-turnover

## Required environment variables

- `DHAN_ACCESS_TOKEN`
- `DHAN_CLIENT_ID`
  - optional if the client id can be decoded from the token

## Local usage

From repo root:

```bash
python dhan_cloud/v9_quant_strategy.py --end-date 2026-04-17
python dhan_cloud/v21_obv5_top2_daily_strategy.py --end-date 2026-04-17
```

Optional:

```bash
python dhan_cloud/v9_quant_strategy.py --output-json cache/dhan_cloud_v9_latest.json
python dhan_cloud/v21_obv5_top2_daily_strategy.py --output-json cache/dhan_cloud_v21_latest.json
```

## Notes

- These wrappers only generate target weights. They do not place orders.
- `v21_obv5_top2_daily` was chosen because it had the highest CAGR in the ETF
  price/volume research, ignoring turnover.
- If you want the safer challenger later, use the slowed weekly
  `mom10 x relative-volume` blend instead of the OBV daily model.
