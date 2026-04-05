# Structured Event Store

This folder holds retrieved event facts for the LLM overlay and critic.

## Layout

- `raw/structured_events.jsonl`
  - append-only event history
- `derived/structured_event_store_active.json`
  - latest active event view
- latest cache mirror:
  - `cache/structured_event_store_latest.json`

## Build / refresh

From repo root:

```bash
./.venv/bin/python -m events.structured_event_store --refresh
```

The event packet is intentionally small:

- official India policy/regulatory feeds when available
- a few focused GDELT geopolitical/macro queries
- compact top-event summaries for the LLM packet

The LLM should use these facts first, and only then use open-web search to confirm or refresh them.
