# Trader Repo Map

This repo is now split into:

- `trader_system/` for the real implementation
- `research/` for experimental branches
- `dashboard/` for the React UI
- `cache/` for generated runtime artifacts
- `deploy/` for schedulers and background services
- `config/` for example files, prompts, and broker maps
- `docs/` for operator docs
- repo root for the repo README and major folders only

## What you actually run

Run commands from the repo root, but run the code out of `trader_system/`:

- `python -m trader_system.runtime.daily_cycle`
- `python -m trader_system.execution.order_planner`
- `python -m trader_system.execution.groww_order_runner`

So:

- `trader_system/` = real code
- repo root = where you stand when you run commands

## Canonical code location

All production code lives under `trader_system/`.
Run it with module commands such as:

- `python -m trader_system.strategy.v9_engine`
- `python -m trader_system.runtime.daily_cycle`
- `python -m trader_system.execution.order_planner`
- `python -m trader_system.execution.groww_order_runner`
- `python -m trader_system.analytics.significance_report`
- `python -m trader_system.analytics.validation_report`

## Implementation folders

- `trader_system/strategy/`
  - main production strategy engine
- `trader_system/broker/`
  - Groww client and instrument map loading
- `trader_system/execution/`
  - rebalance engine, order planning, broker payload adapters, safe order runner
- `trader_system/runtime/`
  - daily cycle, calendar, audit log, runtime store, paper ledger, server helpers
- `trader_system/llm/`
  - Anthropic overlay and overlay-learning logic
- `trader_system/analytics/`
  - significance and validation reports

## Research folder

The experimental branches now live under `research/`:

- `research/alpha_v10_canary_research.py`
- `research/alpha_v11_macro_value_research.py`
- `research/alpha_v12_meta_ensemble.py`
- `research/alpha_v13_sparse_meta.py`
- `research/alpha_v14_sparse_sleeves.py`
- `research/us_portability_v9.py`
- `research/strategy_tester.py`

These are not the main deployment path, but they are still useful reference branches.

## Config and docs

- `config/portfolio_state.example.json`
- `config/groww.env.example`
- `config/groww_universe.example.json`
- `config/groww_universe.production.json`
- `config/llm_overlay_example.json`
- `config/data/README.md`
- `config/prompts/llm_overlay_prompt.md`
- `config/prompts/llm_overlay_meta_prompt.md`
- `docs/RUNBOOK.md`
- `docs/PRODUCTION_CHECKLIST.md`
- `docs/INDIA_DATA_SOURCES.md`
- `docs/strategy_tester.md`

## Tests

- `tests/`

## Generated artifacts

- `cache/`
- `__pycache__/`
- `.pytest_cache/`
- `dashboard/dist/`
- `dashboard/node_modules/`

Treat these as runtime outputs, not hand-maintained source.

## Current recommendation

Use this stack for production prep:

1. `python -m trader_system.strategy.v9_engine --universe-mode benchmark` for research and validation.
2. `python -m trader_system.strategy.v9_engine --universe-mode tradable` only for live sleeve approximation.
3. `python -m trader_system.runtime.daily_cycle` as the one-shot operator command.
4. `python -m trader_system.execution.order_planner` to turn target weights into orders.
5. `python -m trader_system.execution.groww_order_runner` to preview, confirm, place, and reconcile broker orders.
6. `python -m trader_system.llm.anthropic_risk_overlay` only as a sparse risk governor.
7. `dashboard/` as the operator-facing UI.

## What is still left before production

1. Paper trade for several weeks before any live automation.
2. Decide final live operating policy for when `python -m trader_system.execution.groww_order_runner --mode place` is allowed.
3. Add alerting for data failures, Anthropic failures, and reconciliation mismatches.
4. Optionally add fill-level enrichment from broker trade history once live submissions start.

Already implemented:

- Fixed production broker universe in `config/groww_universe.production.json`
- Idempotent daily run logging with preserved realized evaluations in `trader_system/runtime/audit_log.py`
- Safe order workflow in `trader_system/execution/groww_order_runner.py`
- Reconciliation path against live holdings / positions / cash snapshot

## Runtime recommendation

Best architecture:

- `cron` or scheduler for `python -m trader_system.runtime.daily_cycle`
- `dashboard/` as a read-only monitoring UI
- keep order placement separate from the UI

The web app should monitor and inspect. The scheduler should generate decisions. The broker adapter should execute.
