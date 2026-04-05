# Production Checklist

## Before live production

- Stable benchmark and tradable universes defined
- Daily signal generation script stable
- Order planner stable on real holdings snapshots
- Human approval step in place
- Broker submission path tested in paper mode
- Post-trade reconciliation added
- Run log / audit log added
- Monitoring for price failures, Anthropic failures, and empty plans
- Secrets moved out of ad hoc local files and into environment management

## Current status

Done:
- Main strategy engine
- Benchmark mode
- Tradable proxy mode
- Anthropic override harness
- Live signal reporting
- Rebalance planning
- Daily cycle runner
- Validation pack
- Dashboard snapshot path
- Fixed production Groww universe
- Persistent audit log with preserved realized evaluations
- Safe `dry-run` / `confirm` / `place` broker runner
- Reconciliation against live holdings, positions, and available cash

Still needed:
- Daily scheduler around `python -m trader_system.runtime.daily_cycle`
- Alerting
- Shadow-mode track record for the Anthropic overlay
- Fill-level enrichment from broker trades once live placements begin
- Multiple weeks of paper trading before live broker placement

## Recommended runtime

- Cron or scheduled worker runs `python -m trader_system.runtime.daily_cycle`
- Web app is for monitoring and approvals
- Do not make the web app the trading engine

## Deployment recommendation

Phase 1:
- Daily `python -m trader_system.runtime.daily_cycle` run
- Anthropic override only on gated event-risk packets
- Generate dry-run orders
- Human review

Phase 2:
- Paper trading
- Fill and reconciliation logs

Phase 3:
- Limited auto-execution with caps and alerts
