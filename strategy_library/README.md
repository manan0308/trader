# Curated Strategy Library

This folder is the human-readable keep set for the strategies we want to preserve.
It does not duplicate implementation logic; it records the memorable names, original
strategy ids, and the source files that define or assemble each strategy.

## Mapping

| Memorable ID | Human Name | Original Strategy ID | What It Does | Primary Source |
| --- | --- | --- | --- | --- |
| `steady_core_v9` | Steady Core | `v9_quant` | Baseline diversified tactical multi-asset allocator | `strategy/v9_engine.py` |
| `surge_rotation_blend` | Surge Rotation | `v18 70% v9 + 30% momentum` | 70% v9 core plus 30% aggressive relative-strength rotation | `research/alpha_v18_agile_rotation.py` + `research/strategy_matrix_20260419.py` |
| `pullback_guard_blend` | Pullback Guard | `v19 70% v9 + 30% pullback` | 70% v9 core plus 30% trend-safe pullback buyer | `research/alpha_v19_momentum_reversion.py` + `research/strategy_matrix_20260419.py` |
| `polaris_etf_blend` | Polaris ETF Blend | `Shiny ETF v20 fixed spec` | 80% v9 plus 20% conservative pullback blend with smoothing | `research/alpha_v20_deep_blend_search.py` + `research/strategy_matrix_20260419.py` |
| `volume_pulse_weekly` | Volume Pulse Weekly | `v21_mom10_x_vr20_top2_weekly_s050_b020` | Weekly slowed top-2 rotation on 10-day momentum times 20-day relative volume | `research/alpha_v21_price_volume_rotation.py` |
| `flow_pulse_weekly` | Flow Pulse Weekly | `v21_obv5_top2_weekly_s050_b020` | Weekly slowed top-2 rotation on 5-day OBV acceleration | `research/alpha_v21_price_volume_rotation.py` |
| `atlas_jpm_blend` | Atlas JPM Blend | `v24_20v23_jpm_dtf3signal_top3_40v23_jpm_scorecard8_top3_40v9_b025_s100` | JPM-inspired blend of v9, scorecard momentum, and diversified trend following | `research/alpha_v24_jpm_blend_iterative_search.py` + `research/alpha_v23_jpm_momentum_playbook.py` |

## Keep Rules

- Keep the UI and backend layers intact.
- Keep the LLM pipeline and prompts intact.
- Keep the backtest datasets intact.
- Keep the source files required to reconstruct the seven strategies above.
- Do not delete source modules for these strategies until equivalent curated implementations exist.

## Notes

- `Shiny ETF v20 fixed spec` is a named composite assembled in `research/strategy_matrix_20260419.py`.
- The `v18` and `v19` 70/30 blends are also assembled composites, not standalone single-source files.
- `atlas_jpm_blend` depends on both the JPM variant definitions and the v24 blend search helpers.
