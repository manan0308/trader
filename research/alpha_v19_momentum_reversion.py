#!/usr/bin/env python3
"""
ALPHA ENGINE v19 - MOMENTUM + SHORT-TERM MEAN REVERSION
=======================================================

Research-only challenger:
- Keep the medium-term momentum engine that tends to work across assets.
- Add a short-term mean-reversion sleeve that buys pullbacks only when the
  longer trend is still intact.
- Penalize chasing very stretched assets.

The design is intentionally conservative about "mean reversion": it does not
buy broken downtrends just because they are cheap. It buys weakness inside a
still-healthy trend.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
from research.alpha_v12_meta_ensemble import BASE_V9
from research.alpha_v18_agile_rotation import (
    BROAD_SYMBOL_MAP,
    DEFAULT_DHAN,
    DEFAULT_ETF,
    DEFAULT_OVERLAY,
    ETF_SYMBOL_MAP,
    blend_weights,
    cap_and_redistribute,
    fixed_config_wfo,
    load_parquet_prices,
    rotation_summary,
    strip_metrics,
)
from strategy.v9_engine import (
    ALL,
    RISKY,
    CACHE_DIR,
    DEFAULT_RF,
    DEFAULT_TX,
    NarrativeOverlay,
    WARMUP_DAYS,
    breadth_risk_scale,
    build_features,
    crash_signal,
    inverse_vol_weights,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
)


DEFAULT_OUTPUT = CACHE_DIR / "alpha_v19_momentum_reversion_results.json"


@dataclass(frozen=True)
class MomentumReversionConfig:
    name: str
    momentum_weight: float
    reversion_weight: float
    momentum_top_n: int
    reversion_top_n: int
    max_asset_weight: float
    min_cash: float
    weak_breadth_cash_floor: float
    trade_band: float
    trade_step: float
    schedule: str
    min_pullback_z: float
    rsi_ceiling: float
    overextension_z: float
    overextension_penalty: float
    reversion_score_boost: float
    crash_floor: float = 0.70


def candidates() -> List[MomentumReversionConfig]:
    return [
        MomentumReversionConfig(
            name="v19_balanced_pullback",
            momentum_weight=0.65,
            reversion_weight=0.35,
            momentum_top_n=3,
            reversion_top_n=2,
            max_asset_weight=0.38,
            min_cash=0.06,
            weak_breadth_cash_floor=0.25,
            trade_band=0.035,
            trade_step=0.85,
            schedule="WEEKLY",
            min_pullback_z=0.75,
            rsi_ceiling=45.0,
            overextension_z=1.75,
            overextension_penalty=0.04,
            reversion_score_boost=0.08,
        ),
        MomentumReversionConfig(
            name="v19_fast_pullback",
            momentum_weight=0.55,
            reversion_weight=0.45,
            momentum_top_n=3,
            reversion_top_n=2,
            max_asset_weight=0.42,
            min_cash=0.07,
            weak_breadth_cash_floor=0.30,
            trade_band=0.025,
            trade_step=1.00,
            schedule="TWICE_WEEKLY",
            min_pullback_z=0.60,
            rsi_ceiling=48.0,
            overextension_z=1.60,
            overextension_penalty=0.05,
            reversion_score_boost=0.10,
        ),
        MomentumReversionConfig(
            name="v19_conservative_pullback",
            momentum_weight=0.75,
            reversion_weight=0.25,
            momentum_top_n=4,
            reversion_top_n=1,
            max_asset_weight=0.34,
            min_cash=0.08,
            weak_breadth_cash_floor=0.30,
            trade_band=0.045,
            trade_step=0.75,
            schedule="WEEKLY",
            min_pullback_z=1.00,
            rsi_ceiling=42.0,
            overextension_z=2.00,
            overextension_penalty=0.03,
            reversion_score_boost=0.06,
        ),
    ]


def schedule_for_index(index: pd.DatetimeIndex, schedule_name: str) -> pd.Series:
    if schedule_name == "WEEKLY":
        return schedule_flags(index, "WEEKLY")
    if schedule_name == "MONTHLY":
        return schedule_flags(index, "MONTHLY")
    if schedule_name == "TWICE_WEEKLY":
        days = pd.Series(index=index, data=index.weekday)
        return days.isin({2, 4})
    if schedule_name == "DAILY":
        return pd.Series(True, index=index)
    raise ValueError(f"Unsupported schedule: {schedule_name}")


def zscore_cross_section(row: pd.Series) -> pd.Series:
    clean = pd.Series(row).replace([np.inf, -np.inf], np.nan)
    std = float(clean.std())
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=clean.index)
    return ((clean - float(clean.mean())) / std).fillna(0.0)


def run_momentum_reversion_strategy(
    prices: pd.DataFrame,
    config: MomentumReversionConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    features = build_features(prices)
    schedule = schedule_for_index(prices.index, config.schedule)
    override = overlay.risk_off.reindex(prices.index).fillna(0.0).clip(0.0, 1.0) if overlay is not None else pd.Series(0.0, index=prices.index)
    asset_bias = overlay.asset_bias.reindex(prices.index).fillna(0.0) if overlay is not None else pd.DataFrame(0.0, index=prices.index, columns=RISKY)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = pd.DataFrame(features["sma50"])
    sma200 = pd.DataFrame(features["sma200"])
    mom21 = prices[RISKY].pct_change(21, fill_method=None)
    mom63 = pd.DataFrame(features["mom63"])
    mom126 = pd.DataFrame(features["mom126"])
    mom252 = pd.DataFrame(features["mom252"])
    vol20 = prices[RISKY].pct_change(fill_method=None).rolling(20).std() * np.sqrt(21.0)
    vol63 = pd.DataFrame(features["vol63"])
    rsi14 = pd.DataFrame(features["rsi14"])
    asset_dd20 = pd.DataFrame(features["asset_dd20"])
    cash126 = pd.Series(features["cash126"])

    ret5 = prices[RISKY].pct_change(5, fill_method=None)
    ret10 = prices[RISKY].pct_change(10, fill_method=None)
    pullback_z = (-(0.60 * ret5 + 0.40 * ret10) / vol20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    extension_z = ((0.60 * ret5 + 0.40 * ret10) / vol20.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    for i, dt in enumerate(prices.index):
        if i < WARMUP_DAYS:
            weights.iloc[i] = current
            continue

        trend = (prices.loc[dt, RISKY] > sma200.loc[dt]) | (mom126.loc[dt] > cash126.loc[dt])
        healthy_pullback = (
            trend
            & (prices.loc[dt, RISKY] > sma50.loc[dt])
            & (pullback_z.loc[dt] >= config.min_pullback_z)
            & (rsi14.loc[dt] <= config.rsi_ceiling)
            & (asset_dd20.loc[dt] > -0.16)
        ).fillna(False)
        eligible_momentum = (trend & (mom63.loc[dt] > -0.04)).fillna(False)
        crash = crash_signal(features, dt)

        if bool(schedule.loc[dt]) or bool(healthy_pullback.any()):
            breadth = int(eligible_momentum.sum())
            risk_scale = breadth_risk_scale(breadth)
            if breadth <= 1:
                risk_scale = min(risk_scale, 1.0 - config.weak_breadth_cash_floor)
            if crash:
                risk_scale = min(risk_scale, config.crash_floor)

            llm_override = float(override.loc[dt])
            risk_scale = min(risk_scale, 1.0 - 0.65 * llm_override)
            risk_scale = min(risk_scale, 1.0 - config.min_cash)

            momentum_raw = (
                0.35 * zscore_cross_section(mom63.loc[dt])
                + 0.35 * zscore_cross_section(mom126.loc[dt] - cash126.loc[dt])
                + 0.20 * zscore_cross_section(mom252.loc[dt])
                + 0.10 * zscore_cross_section(mom63.loc[dt] / vol63.loc[dt].replace(0.0, np.nan))
            )
            momentum_raw = momentum_raw.reindex(RISKY).fillna(-999.0)
            extended = (extension_z.loc[dt] > config.overextension_z).fillna(False)
            momentum_raw = momentum_raw - extended.astype(float) * config.overextension_penalty

            reversion_raw = (
                pullback_z.loc[dt].reindex(RISKY).fillna(0.0)
                + (config.rsi_ceiling - rsi14.loc[dt].reindex(RISKY).fillna(50.0)).clip(lower=0.0) / 20.0
                + (-asset_dd20.loc[dt].reindex(RISKY).fillna(0.0)).clip(lower=0.0, upper=0.12) * 4.0
            )
            reversion_raw = reversion_raw + healthy_pullback.astype(float) * config.reversion_score_boost

            momentum_ranked = momentum_raw.sort_values(ascending=False)
            momentum_assets = [asset for asset in momentum_ranked.index if bool(eligible_momentum.get(asset, False))][: config.momentum_top_n]

            reversion_ranked = reversion_raw.sort_values(ascending=False)
            reversion_assets = [asset for asset in reversion_ranked.index if bool(healthy_pullback.get(asset, False))][: config.reversion_top_n]

            target = pd.Series(0.0, index=ALL)
            if momentum_assets:
                mom_weights = inverse_vol_weights(vol63.loc[dt], momentum_assets)
                mom_weights = cap_and_redistribute(mom_weights, config.max_asset_weight)
                momentum_budget = config.momentum_weight + (config.reversion_weight if not reversion_assets else 0.0)
                for asset, weight in mom_weights.items():
                    target[asset] += risk_scale * momentum_budget * weight

            if reversion_assets:
                rev_scores = reversion_ranked.loc[reversion_assets].clip(lower=0.0)
                if rev_scores.sum() <= 0:
                    rev_scores[:] = 1.0
                rev_weights = cap_and_redistribute(rev_scores / rev_scores.sum(), config.max_asset_weight)
                for asset, weight in rev_weights.items():
                    target[asset] += risk_scale * config.reversion_weight * weight

            if target[RISKY].sum() > 0:
                risky = cap_and_redistribute(target[RISKY] / target[RISKY].sum(), config.max_asset_weight)
                target[RISKY] = risky * float(target[RISKY].sum())

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            if target[RISKY].sum() > 0 and float(bias_row.abs().sum()) > 0:
                multipliers = 1.0 + 0.15 * bias_row
                adjusted = target[RISKY] * multipliers
                if adjusted.sum() > 0:
                    target[RISKY] = adjusted / adjusted.sum() * target[RISKY].sum()

            target["CASH"] = max(0.0, 1.0 - target[RISKY].sum())
            if target.sum() > 0:
                target = target / target.sum()

            proposal = current * (1.0 - config.trade_step) + target * config.trade_step
            proposal["CASH"] = max(0.0, 1.0 - proposal[RISKY].sum())
            proposal = proposal / proposal.sum()
            if float((proposal - current).abs().max()) > config.trade_band:
                current = proposal

        if crash and current[RISKY].sum() > 0.80:
            current = current.copy()
            current[RISKY] *= 0.90
            current["CASH"] = 1.0 - current[RISKY].sum()

        llm_live_risk = float(override.loc[dt])
        if llm_live_risk > 0.15 and current[RISKY].sum() > 0:
            min_cash = float(np.clip(0.20 + 0.45 * llm_live_risk, 0.20, 0.70))
            max_risky = 1.0 - min_cash
            risky_sum = float(current[RISKY].sum())
            if risky_sum > max_risky:
                current = current.copy()
                current[RISKY] *= max_risky / risky_sum
                current["CASH"] = 1.0 - current[RISKY].sum()

        weights.iloc[i] = current

    return weights


def slice_overlay(overlay: Optional[NarrativeOverlay], index: pd.DatetimeIndex) -> Optional[NarrativeOverlay]:
    if overlay is None:
        return None
    return NarrativeOverlay(
        risk_off=overlay.risk_off.reindex(index).fillna(0.0),
        asset_bias=overlay.asset_bias.reindex(index).fillna(0.0),
    )


def run_dataset(
    name: str,
    prices: pd.DataFrame,
    overlay: Optional[NarrativeOverlay],
    rf: float,
    tx_cost: float,
    train_days: int,
    test_days: int,
    cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
) -> Dict[str, object]:
    dataset_overlay = slice_overlay(overlay, prices.index) if overlay is not None else None
    overlay_active_days = int((dataset_overlay.risk_off > 0).sum()) if dataset_overlay is not None else 0

    runners: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "v9_quant": lambda frame: run_strategy(frame, BASE_V9, overlay=None),
    }
    cfgs = candidates()
    for cfg in cfgs:
        runners[cfg.name] = lambda frame, cfg=cfg: run_momentum_reversion_strategy(frame, cfg, overlay=None)

    cfg_by_name = {cfg.name: cfg for cfg in cfgs}
    for cfg_name in ("v19_balanced_pullback", "v19_fast_pullback"):
        cfg = cfg_by_name[cfg_name]
        runners[f"v19_blend_70v9_30_{cfg_name.replace('v19_', '')}"] = lambda frame, cfg=cfg: blend_weights(
            run_strategy(frame, BASE_V9, overlay=None),
            run_momentum_reversion_strategy(frame, cfg, overlay=None),
            0.70,
        )
        runners[f"v19_blend_50v9_50_{cfg_name.replace('v19_', '')}"] = lambda frame, cfg=cfg: blend_weights(
            run_strategy(frame, BASE_V9, overlay=None),
            run_momentum_reversion_strategy(frame, cfg, overlay=None),
            0.50,
        )

    if dataset_overlay is not None and overlay_active_days > 0:
        runners["v9_quant_plus_llm"] = lambda frame: run_strategy(frame, BASE_V9, overlay=slice_overlay(dataset_overlay, frame.index))

    full_rows: List[Dict[str, object]] = []
    full_payload: Dict[str, object] = {}
    weight_summaries: Dict[str, object] = {}
    for label, runner in runners.items():
        weights = runner(prices)
        metrics = performance_metrics(
            prices,
            weights,
            label,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )
        full_rows.append(metrics)
        full_payload[label] = strip_metrics(metrics)
        weight_summaries[label] = rotation_summary(weights)
    print_results_table(f"FULL SAMPLE - {name}", full_rows)

    wfo_payload: Dict[str, object] = {}
    wfo_rows: List[Dict[str, object]] = []
    for label, runner in runners.items():
        payload = fixed_config_wfo(
            prices,
            runner,
            label,
            rf=rf,
            tx_cost=tx_cost,
            train_days=train_days,
            test_days=test_days,
            cost_model=cost_model,
            base_value=base_value,
        )
        wfo_rows.append(payload["metrics"])
        wfo_payload[label] = {
            "metrics": strip_metrics(payload["metrics"]),
            "windows": payload["windows"],
        }
    print_results_table(f"WALK-FORWARD OOS - {name}", wfo_rows)

    return {
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "rows": int(len(prices)),
        "overlay_active_days": overlay_active_days,
        "train_days": train_days,
        "test_days": test_days,
        "full_sample": full_payload,
        "walk_forward": wfo_payload,
        "weight_summaries": weight_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test momentum + short-term mean-reversion pullback challengers.")
    parser.add_argument("--dhan-parquet", default=str(DEFAULT_DHAN))
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--overlay-file", default=str(DEFAULT_OVERLAY))
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--cost-model", choices=["flat", "india_delivery"], default="india_delivery")
    parser.add_argument("--base-value", type=float, default=1_000_000.0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    tx_cost = args.tx_bps / 10_000.0
    cost_model = resolve_cost_model(args.cost_model)
    output_path = Path(args.output).expanduser().resolve()
    overlay_path = Path(args.overlay_file).expanduser().resolve() if args.overlay_file else None

    dhan_prices = load_parquet_prices(Path(args.dhan_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    overlay = load_llm_overlay(str(overlay_path), dhan_prices.index) if overlay_path and overlay_path.exists() else None

    datasets: Dict[str, object] = {
        "dhan_fy14_fy26": run_dataset(
            "Dhan FY14-FY26",
            dhan_prices,
            overlay,
            rf=args.rf,
            tx_cost=tx_cost,
            train_days=756,
            test_days=126,
            cost_model=cost_model,
            base_value=args.base_value,
        )
    }

    etf_path = Path(args.etf_parquet).expanduser().resolve()
    if etf_path.exists():
        etf_prices = load_parquet_prices(etf_path, ETF_SYMBOL_MAP)
        etf_overlay = load_llm_overlay(str(overlay_path), etf_prices.index) if overlay_path and overlay_path.exists() else None
        datasets["etf_fy24_fy26"] = run_dataset(
            "Dhan ETF FY24-FY26",
            etf_prices,
            etf_overlay,
            rf=args.rf,
            tx_cost=tx_cost,
            train_days=252,
            test_days=63,
            cost_model=cost_model,
            base_value=args.base_value,
        )

    payload = {
        "tx_cost": tx_cost,
        "cost_model": args.cost_model,
        "base_value": args.base_value,
        "rf": args.rf,
        "overlay_file": str(overlay_path) if overlay_path else None,
        "literature_shape": [
            "Medium-term momentum: rank assets by 3-12 month strength.",
            "Short-term reversal: buy temporary 5-10 day weakness only inside intact trends.",
            "Avoid falling knives: require SMA/trend health and cap drawdown.",
            "Avoid chasing: penalize assets after very strong 5-10 day extensions.",
        ],
        "configs": {cfg.name: asdict(cfg) for cfg in candidates()},
        "datasets": datasets,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
