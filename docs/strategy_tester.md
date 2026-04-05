# Strategy Tester

Run a strategy file against the shared cleaned INR universe:

```bash
python strategy_tester.py --strategy strategy_example_stub.py
```

What the tester looks for:

- Price fetch functions: `fetch_data`, `fetch`, `fetch_universe_prices`, `load_prices`
- Strategy functions: `run_strategy`, `generate_signals`, `tactical_weights`, `weights`, `backtest`, `bt`

Minimal contract for a strategy script:

- Return a `pandas.DataFrame` of weights indexed by date.
- Use the standard columns: `NIFTY`, `MIDCAP`, `SMALLCAP`, `GOLD`, `SILVER`, `US`, `CASH`.
- If you already have a `backtest` function, the tester can use that too.

Examples:

```bash
python strategy_tester.py --strategy my_strategy.py
python strategy_tester.py --strategy my_strategy.py --strategy-fn run_strategy
python strategy_tester.py --strategy my_strategy.py --params '{"tilt": 0.15}'
python strategy_tester.py --strategy my_strategy.py --force-standard-data
python strategy_tester.py my_strategy.py
```

Outputs:

- Full-sample metrics
- Standard benchmark comparison
- Optional walk-forward OOS summary
- JSON summary via `--output-json`
