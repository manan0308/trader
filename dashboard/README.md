# Trader Dashboard

Lightweight React dashboard for the alpha engine snapshots.

## Run

```bash
cd dashboard
npm install
npm run sync
npm run dev
```

## Build

```bash
cd dashboard
npm run build
```

## Serve The Built App

```bash
cd ..
./.venv/bin/python -m runtime.local_api_server --host 127.0.0.1 --port 8050
```

Open:

- `http://127.0.0.1:8050/`
- `http://127.0.0.1:8050/api`

## UI layout

The dashboard is intentionally tabbed now:

- `V9`: main operating screen with current v9 weights, paper equity, and recent v9 signals
- `Paper`: paper account curve and current paper positions
- `Orders`: latest dry-run execution plan
- `Research`: benchmark and validation views
- `LLM`: overlay state and learning summary
- `Setup`: background-service install commands and quick operational help

## Data contract

The app now prefers `GET /api/dashboard` from the Python server and falls back to `public/data/dashboard.json` if
needed. The snapshot file is generated from the Python cache files by `scripts/sync-data.mjs`.

It expects optional `paper_trading`, `paper_base`, `paper_comparison`, `audit_runs`, `learning_state`, `overlay_raw`,
`overlay_active`, and `daily_cycle` sections in that snapshot. Missing sections are handled gracefully and render as
empty states.

Refresh the underlying live snapshot first if you want current signals:

```bash
./.venv/bin/python -m runtime.daily_cycle --skip-overlay
```

To make it run in the background on macOS, use the install helper from the repo root:

```bash
./deploy/bin/install_launchd.sh
```
