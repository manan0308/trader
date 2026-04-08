# Trader Runbook

> **Repo path.** All commands below assume you have `cd`'d into the repo
> root (the directory that contains this `docs/` folder). Set
> `TRADER_ROOT=$(pwd)` once and reuse it from new shells:
>
> ```bash
> export TRADER_ROOT=/absolute/path/to/trader   # adjust for your machine
> cd "$TRADER_ROOT"
> ```

## Main daily refresh

Run the full signal -> learning -> execution-plan -> dashboard sync cycle:

```bash
cd "$TRADER_ROOT"
./.venv/bin/python -m runtime.daily_cycle --portfolio-file config/portfolio_state.example.json
```

By default the live daily cycle now prefers `Groww` for the signal/execution path when broker auth is healthy, and falls back to `yfinance` otherwise.

Use `--skip-overlay` to avoid Anthropic calls:

```bash
./.venv/bin/python -m runtime.daily_cycle --skip-overlay --portfolio-file config/portfolio_state.example.json
```

Reset paper, audit, and learning state:

```bash
./.venv/bin/python -m runtime.daily_cycle --skip-overlay --reset-state --portfolio-file config/portfolio_state.example.json
```

## Groww auth bootstrap

Groww broker auth is treated as a morning bootstrap step on the VPS. The clean path is:

1. whitelist the VPS public static IPv4 with Groww
2. set `GROWW_API_KEY` plus either `GROWW_TOTP_SECRET` or `GROWW_API_SECRET`
3. refresh the broker token before market open

Manual refresh:

```bash
./.venv/bin/python -m runtime.groww_auth_refresh
```

This writes:

- `config/runtime.env` with the fresh access token
- `cache/groww_auth_status.json` for the API and dashboard

Check broker auth status:

```bash
cat cache/groww_auth_status.json
```

## Exact weights on any date

Ask for exact model weights on any date. If the date is a holiday or weekend, it resolves to the latest previous trading day.

```bash
./.venv/bin/python -m runtime.weights_on_date --model v9 --date 2026-04-02
./.venv/bin/python -m runtime.weights_on_date --model v9 --date 2026-04-03 --json
```

## Paper backfill from a start date

Initialize the paper book with `₹10,00,000` from `2026-04-01` and carry it forward day by day:

```bash
./.venv/bin/python -m runtime.paper_backfill --start 2026-04-01 --end 2026-04-03 --initial-cash 1000000 --reset --apply
```

This writes:

- `cache/paper_backfill_latest.json`
- `cache/paper_trading_state.json`
- `cache/paper_trading_latest.json`
- `cache/paper_trading_journal.jsonl`

## Dashboard UI

Build the React app:

```bash
cd "$TRADER_ROOT/dashboard"
npm install
npm run build
```

Serve the built dashboard and JSON APIs:

```bash
cd "$TRADER_ROOT"
./.venv/bin/python -m runtime.local_api_server --host 127.0.0.1 --port 8050
```

Open in browser:

- `http://127.0.0.1:8050/`

## JSON endpoints

- `http://127.0.0.1:8050/api`
- `http://127.0.0.1:8050/api/dashboard`
- `http://127.0.0.1:8050/api/live_signal`
- `http://127.0.0.1:8050/api/execution_plan`
- `http://127.0.0.1:8050/api/overlay`
- `http://127.0.0.1:8050/api/overlay_raw`
- `http://127.0.0.1:8050/api/validation`
- `http://127.0.0.1:8050/api/learning_state`
- `http://127.0.0.1:8050/api/paper_trading`
- `http://127.0.0.1:8050/api/paper_backfill`
- `http://127.0.0.1:8050/api/audit_runs`
- `http://127.0.0.1:8050/api/daily_cycle`
- `http://127.0.0.1:8050/api/execution_submissions`
- `http://127.0.0.1:8050/api/execution_confirmation`
- `http://127.0.0.1:8050/api/reconciliation`

## One-off utilities

Recent live-style signals:

```bash
./.venv/bin/python -m runtime.live_signal_report --model all --days 5 --universe-mode benchmark
```

Dry-run execution plan:

```bash
./.venv/bin/python -m execution.order_planner --portfolio-file config/portfolio_state.example.json --universe-mode tradable
```

Safe broker-facing workflow:

```bash
./.venv/bin/python -m execution.groww_order_runner --mode dry-run
./.venv/bin/python -m execution.groww_order_runner --mode confirm --note "Reviewed latest v9 execution plan"
./.venv/bin/python -m execution.groww_order_runner --mode place
./.venv/bin/python -m execution.groww_order_runner --mode reconcile
```

Notes:

- `dry-run` persists the exact order payload preview.
- `confirm` records operator approval for the current `plan_key`.
- `place` requires confirmation and uses deterministic reference IDs to avoid duplicate placement.
- `reconcile` compares target quantities and weights against live broker holdings, positions, and cash.
- The default broker map is `config/groww_universe.production.json`.

Validation pack:

```bash
./.venv/bin/python -m analytics.validation_report
```

Significance pack:

```bash
./.venv/bin/python -m analytics.significance_report
```

## Background services on macOS

Install and start the background jobs:

```bash
chmod +x deploy/bin/install_launchd.sh deploy/bin/uninstall_launchd.sh
./deploy/bin/install_launchd.sh
```

This enables:

- `com.trader.dashboard`: always-on local dashboard/API server on `127.0.0.1:8050`
- `com.trader.daily-cycle`: scheduled run at `18:45` IST every trading day

View status:

```bash
launchctl list | rg 'com\\.trader\\.(daily-cycle|dashboard)'
```

Remove the jobs:

```bash
./deploy/bin/uninstall_launchd.sh
```

## Background services on Linux

Install the service files from `deploy/systemd/` into `~/.config/systemd/user/` or `/etc/systemd/system/`, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now trader-dashboard.service
systemctl --user enable --now trader-daily-cycle.timer
systemctl --user enable --now trader-groww-auth.timer
```

Recommended cadence:

- `trader-groww-auth.timer`: `08:35 IST`
- `trader-daily-cycle.timer`: `18:45 IST`
