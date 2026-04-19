#!/usr/bin/env python3
"""
ALPHA ENGINE v18 — AGILE RELATIVE-STRENGTH ROTATION
===================================================

Research-only challenger for the core issue we saw in v9 weight charts:
v9 is a stable multi-asset allocator, not a true alpha rotator.

This script tests variants that:
- shrink the always-on broad core,
- allocate core only to assets that are actually in trend / beating cash,
- put a larger share into the strongest ranked assets,
- let weak assets fall to zero instead of living inside the permanent core.

Promotion rule:
Do not ship just because this looks more exciting. Promote only if stitched
OOS improves Sharpe / CAGR without unacceptable drawdown and turnover.
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

from research.alpha_v12_meta_ensemble import BASE_V9
from execution.india_costs import IndianDeliveryCostModel, resolve_cost_model
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
    format_metrics,
    inverse_vol_weights,
    load_llm_overlay,
    performance_metrics,
    print_results_table,
    run_strategy,
    schedule_flags,
)


DEFAULT_DHAN = Path("/Users/mananagarwal/Desktop/historical data/dhan/all_daily_fy14_fy26_aligned.parquet")
DEFAULT_ETF = Path("/Users/mananagarwal/Desktop/historical data/dhan/etfs/all_daily_fy24_fy26_aligned.parquet")
DEFAULT_OUTPUT = CACHE_DIR / "alpha_v18_agile_rotation_results.json"
DEFAULT_OVERLAY = CACHE_DIR / "overlay_archive_from_audit_active_signal_days.json"

BROAD_SYMBOL_MAP = {
    "NIFTY50": "NIFTY",
    "NIFTY_MIDCAP150": "MIDCAP",
    "NIFTY_SMLCAP250": "SMALLCAP",
    "GOLDBEES": "GOLD",
    "XAGINR": "SILVER",
    "MON100": "US",
    "LIQUIDBEES": "CASH",
}
ETF_SYMBOL_MAP = {
    "NIFTYBEES": "NIFTY",
    "MID150BEES": "MIDCAP",
    "HDFCSML250": "SMALLCAP",
    "GOLDBEES": "GOLD",
    "SILVERBEES": "SILVER",
    "MON100": "US",
    "LIQUIDBEES": "CASH",
}


@dataclass(frozen=True)
class AgileRotationConfig:
    name: str
    core_weight: float
    tilt_weight: float
    core_top_n: int
    tilt_top_n: int
    trade_band: float
    trade_step: float
    max_asset_weight: float
    min_cash: float
    require_positive_63d: bool
    require_above_cash_126d: bool
    weak_breadth_cash_floor: float
    crash_floor: float = 0.70
    rebound_bonus: float = 0.03
    breakout_bonus: float = 0.03


def agile_candidates() -> List[AgileRotationConfig]:
    return [
        AgileRotationConfig(
            name="v18_balanced_rotation",
            core_weight=0.60,
            tilt_weight=0.40,
            core_top_n=4,
            tilt_top_n=3,
            trade_band=0.05,
            trade_step=0.90,
            max_asset_weight=0.34,
            min_cash=0.03,
            require_positive_63d=False,
            require_above_cash_126d=True,
            weak_breadth_cash_floor=0.20,
        ),
        AgileRotationConfig(
            name="v18_aggressive_rotation",
            core_weight=0.45,
            tilt_weight=0.55,
            core_top_n=3,
            tilt_top_n=2,
            trade_band=0.035,
            trade_step=1.00,
            max_asset_weight=0.38,
            min_cash=0.04,
            require_positive_63d=True,
            require_above_cash_126d=True,
            weak_breadth_cash_floor=0.25,
        ),
        AgileRotationConfig(
            name="v18_momentum_heavy_rotation",
            core_weight=0.30,
            tilt_weight=0.70,
            core_top_n=3,
            tilt_top_n=2,
            trade_band=0.025,
            trade_step=1.00,
            max_asset_weight=0.45,
            min_cash=0.05,
            require_positive_63d=True,
            require_above_cash_126d=True,
            weak_breadth_cash_floor=0.30,
        ),
    ]


def blend_weights(left: pd.DataFrame, right: pd.DataFrame, left_weight: float) -> pd.DataFrame:
    aligned_left = left.reindex(columns=ALL).fillna(0.0)
    aligned_right = right.reindex(index=aligned_left.index, columns=ALL).fillna(0.0)
    blended = left_weight * aligned_left + (1.0 - left_weight) * aligned_right
    row_sum = blended.sum(axis=1).replace(0.0, np.nan)
    return blended.div(row_sum, axis=0).fillna(0.0)


def load_parquet_prices(path: Path, symbol_map: Dict[str, str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame = frame[frame["symbol"].isin(symbol_map)].copy()
    frame["asset"] = frame["symbol"].map(symbol_map)
    frame["date"] = pd.to_datetime(frame["date"])
    return (
        frame.pivot(index="date", columns="asset", values="close")
        .sort_index()
        .reindex(columns=ALL)
        .dropna()
    )


def cap_and_redistribute(weights: pd.Series, cap: float) -> pd.Series:
    out = weights.copy().clip(lower=0.0)
    if out.sum() <= 0:
        return out
    out = out / out.sum()
    for _ in range(10):
        over = out > cap
        if not bool(over.any()):
            break
        excess = float((out[over] - cap).sum())
        out[over] = cap
        under = ~over
        under_sum = float(out[under].sum())
        if under_sum <= 0 or excess <= 0:
            break
        out[under] += out[under] / under_sum * excess
    return out / out.sum() if out.sum() > 0 else out


def slice_overlay(overlay: Optional[NarrativeOverlay], index: pd.DatetimeIndex) -> Optional[NarrativeOverlay]:
    if overlay is None:
        return None
    return NarrativeOverlay(
        risk_off=overlay.risk_off.reindex(index).fillna(0.0),
        asset_bias=overlay.asset_bias.reindex(index).fillna(0.0),
    )


def run_agile_rotation_strategy(
    prices: pd.DataFrame,
    config: AgileRotationConfig,
    overlay: Optional[NarrativeOverlay] = None,
) -> pd.DataFrame:
    features = build_features(prices)
    schedule = schedule_flags(prices.index, "WEEKLY")
    override = overlay.risk_off.reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0) if overlay is not None else pd.Series(0.0, index=prices.index)
    asset_bias = overlay.asset_bias.reindex(prices.index).fillna(0.0) if overlay is not None else pd.DataFrame(0.0, index=prices.index, columns=RISKY)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    sma50 = pd.DataFrame(features["sma50"])
    sma200 = pd.DataFrame(features["sma200"])
    mom63 = pd.DataFrame(features["mom63"])
    mom126 = pd.DataFrame(features["mom126"])
    mom252 = pd.DataFrame(features["mom252"])
    vol63 = pd.DataFrame(features["vol63"])
    rsi14 = pd.DataFrame(features["rsi14"])
    asset_dd20 = pd.DataFrame(features["asset_dd20"])
    asset_vol_ratio = pd.DataFrame(features["asset_vol_ratio"])
    breakout252 = pd.DataFrame(features["breakout252"])
    cash126 = pd.Series(features["cash126"])

    for i, dt in enumerate(prices.index):
        if i < WARMUP_DAYS:
            weights.iloc[i] = current
            continue

        trend = prices.loc[dt, RISKY] > sma200.loc[dt]
        strong = trend & (prices.loc[dt, RISKY] > sma50.loc[dt]) & (mom126.loc[dt] > cash126.loc[dt])
        crash = crash_signal(features, dt)

        if bool(schedule.loc[dt]):
            eligible = trend.copy()
            if config.require_above_cash_126d:
                eligible = eligible & (mom126.loc[dt] > cash126.loc[dt])
            if config.require_positive_63d:
                eligible = eligible & (mom63.loc[dt] > 0.0)
            eligible = eligible.fillna(False)

            breadth = int(eligible.sum())
            risk_scale = breadth_risk_scale(breadth)
            if breadth <= 1:
                risk_scale = min(risk_scale, 1.0 - config.weak_breadth_cash_floor)
            if crash:
                risk_scale = min(risk_scale, config.crash_floor)

            llm_override = float(override.loc[dt])
            risk_scale = min(risk_scale, 1.0 - 0.65 * llm_override)
            risk_scale = min(risk_scale, 1.0 - config.min_cash)

            score = (
                0.30 * (mom63.loc[dt] / vol63.loc[dt].replace(0, np.nan))
                + 0.35 * mom126.loc[dt]
                + 0.35 * mom252.loc[dt]
            )
            score = score.replace([np.inf, -np.inf], np.nan).fillna(-999.0)

            rebound_setup = trend & (asset_dd20.loc[dt] <= -0.08) & (rsi14.loc[dt] <= 45.0)
            score = score + rebound_setup.astype(float) * config.rebound_bonus
            for asset in ("SILVER", "GOLD", "US"):
                if bool(breakout252.loc[dt].get(asset, False)) and bool(trend.get(asset, False)):
                    score.loc[asset] = float(score.get(asset, -999.0)) + config.breakout_bonus

            ranked = score.sort_values(ascending=False)
            core_assets = [asset for asset in ranked.index if bool(eligible.get(asset, False))][: config.core_top_n]
            tilt_assets = [asset for asset in ranked.index if bool(eligible.get(asset, False))][: config.tilt_top_n]

            target = pd.Series(0.0, index=ALL)
            if core_assets:
                core_w = inverse_vol_weights(vol63.loc[dt], core_assets)
                core_w = cap_and_redistribute(core_w, config.max_asset_weight)
                for asset, weight in core_w.items():
                    target[asset] += risk_scale * config.core_weight * weight

            if tilt_assets and config.tilt_weight > 0:
                tilt_scores = ranked.loc[tilt_assets].clip(lower=0.0)
                if tilt_scores.sum() <= 0:
                    tilt_scores[:] = 1.0
                tilt_scores = cap_and_redistribute(tilt_scores / tilt_scores.sum(), config.max_asset_weight)
                for asset, weight in tilt_scores.items():
                    target[asset] += risk_scale * config.tilt_weight * weight

            if target[RISKY].sum() > 0:
                risky = cap_and_redistribute(target[RISKY] / target[RISKY].sum(), config.max_asset_weight)
                target[RISKY] = risky * float(target[RISKY].sum())

            if crash and bool(trend.get("GOLD", False)):
                room = max(0.0, 1.0 - target[RISKY].sum() - config.min_cash)
                target["GOLD"] += min(0.06, room)

            bias_row = asset_bias.loc[dt].reindex(RISKY).clip(lower=-1.0, upper=1.0)
            if target[RISKY].sum() > 0 and float(bias_row.abs().sum()) > 0:
                multipliers = 1.0 + 0.15 * bias_row
                adjusted = target[RISKY] * multipliers
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum() * target[RISKY].sum()
                    target[RISKY] = adjusted

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


def fixed_config_wfo(
    prices: pd.DataFrame,
    runner: Callable[[pd.DataFrame], pd.DataFrame],
    label: str,
    rf: float,
    tx_cost: float,
    train_days: int,
    test_days: int,
    cost_model: Optional[IndianDeliveryCostModel] = None,
    base_value: float = 1_000_000.0,
) -> Dict[str, object]:
    windows = []
    cursor = 0
    while cursor + train_days + test_days <= len(prices):
        windows.append((cursor, cursor + train_days, cursor + train_days + test_days))
        cursor += test_days

    if not windows:
        empty_returns = pd.Series(dtype=float)
        empty_weights = pd.DataFrame(columns=ALL, dtype=float)
        return {
            "windows": [],
            "metrics": {
                "label": label,
                "weights": empty_weights,
                "returns": empty_returns,
                "equity": pd.Series(dtype=float),
                "cagr": np.nan,
                "vol": np.nan,
                "sharpe": np.nan,
                "mdd": np.nan,
                "calmar": np.nan,
                "turnover": np.nan,
                "avg_cash": np.nan,
            },
        }

    stitched_returns: List[pd.Series] = []
    stitched_weights: List[pd.DataFrame] = []
    rows: List[Dict[str, object]] = []
    for window_id, (start_i, mid_i, end_i) in enumerate(windows, start=1):
        combined_prices = prices.iloc[start_i:end_i]
        test_prices = prices.iloc[mid_i:end_i]
        combined_weights = runner(combined_prices)
        test_weights = combined_weights.loc[test_prices.index]
        test_metrics = performance_metrics(
            test_prices,
            test_weights,
            label,
            rf=rf,
            tx_cost=tx_cost,
            cost_model=cost_model,
            base_value=base_value,
        )
        rows.append(
            {
                "window": window_id,
                "test_start": test_prices.index[0].strftime("%Y-%m-%d"),
                "test_end": test_prices.index[-1].strftime("%Y-%m-%d"),
                "cagr": float(test_metrics["cagr"]),
                "sharpe": float(test_metrics["sharpe"]),
                "mdd": float(test_metrics["mdd"]),
            }
        )
        stitched_returns.append(pd.Series(test_metrics["returns"]))
        stitched_weights.append(test_weights)

    stitched = pd.concat(stitched_returns).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    weights = pd.concat(stitched_weights).sort_index()
    weights = weights[~weights.index.duplicated(keep="last")]
    weights = weights.reindex(stitched.index).ffill().fillna(0.0)

    equity = (1.0 + stitched).cumprod()
    years = len(stitched) / 252.0
    rf_daily = (1.0 + rf) ** (1.0 / 252.0) - 1.0
    excess = stitched - rf_daily
    vol = stitched.std() * np.sqrt(252.0)
    mdd = (equity / equity.cummax() - 1.0).min()
    turnover = (weights.diff().abs().sum(axis=1) / 2.0).sum() / years

    return {
        "windows": rows,
        "metrics": {
            "label": label,
            "weights": weights,
            "returns": stitched,
            "equity": equity,
            "cagr": equity.iloc[-1] ** (1.0 / years) - 1.0,
            "vol": vol,
            "sharpe": excess.mean() * 252.0 / vol if vol > 0 else np.nan,
            "mdd": mdd,
            "calmar": (equity.iloc[-1] ** (1.0 / years) - 1.0) / abs(mdd) if mdd < 0 else np.nan,
            "turnover": turnover,
            "avg_cash": weights["CASH"].mean(),
        },
    }


def strip_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    clean = {
        key: float(value)
        for key, value in metrics.items()
        if key not in {"label", "weights", "returns", "equity"}
    }
    clean["pretty"] = format_metrics(metrics)
    return clean


def rotation_summary(weights: pd.DataFrame) -> Dict[str, object]:
    changes = weights.diff().abs().sum(axis=1) / 2.0
    active = changes[changes > 1e-6]
    return {
        "change_days": int(len(active)),
        "avg_half_turnover_on_change_days": float(active.mean()) if len(active) else 0.0,
        "median_half_turnover_on_change_days": float(active.median()) if len(active) else 0.0,
        "avg_weights": {asset: float(weights[asset].mean()) for asset in ALL},
        "max_weights": {asset: float(weights[asset].max()) for asset in ALL},
        "last_weights": {asset: float(weights[asset].iloc[-1]) for asset in ALL},
    }


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
    candidates = agile_candidates()
    candidate_by_name = {cfg.name: cfg for cfg in candidates}
    for cfg in candidates:
        runners[cfg.name] = lambda frame, cfg=cfg: run_agile_rotation_strategy(frame, cfg, overlay=None)
    for cfg_name in ("v18_aggressive_rotation", "v18_momentum_heavy_rotation"):
        cfg = candidate_by_name[cfg_name]
        runners[f"v18_blend_70v9_30_{cfg_name.replace('v18_', '')}"] = lambda frame, cfg=cfg: blend_weights(
            run_strategy(frame, BASE_V9, overlay=None),
            run_agile_rotation_strategy(frame, cfg, overlay=None),
            0.70,
        )
        runners[f"v18_blend_50v9_50_{cfg_name.replace('v18_', '')}"] = lambda frame, cfg=cfg: blend_weights(
            run_strategy(frame, BASE_V9, overlay=None),
            run_agile_rotation_strategy(frame, cfg, overlay=None),
            0.50,
        )

    if dataset_overlay is not None and overlay_active_days > 0:
        runners["v9_quant_plus_llm"] = lambda frame: run_strategy(frame, BASE_V9, overlay=slice_overlay(dataset_overlay, frame.index))
        for cfg in candidates:
            runners[f"{cfg.name}_plus_llm"] = lambda frame, cfg=cfg: run_agile_rotation_strategy(
                frame,
                cfg,
                overlay=slice_overlay(dataset_overlay, frame.index),
            )

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
    parser = argparse.ArgumentParser(description="Test agile rotation challengers against v9.")
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
    overlay_path = Path(args.overlay_file).expanduser().resolve() if args.overlay_file else None
    output_path = Path(args.output).expanduser().resolve()
    overlay = None

    # Load against the broad dataset index first if possible; individual dataset
    # runs slice the overlay down to their own date range.
    dhan_prices = load_parquet_prices(Path(args.dhan_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    if overlay_path and overlay_path.exists():
        overlay = load_llm_overlay(str(overlay_path), dhan_prices.index)

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
        "configs": {cfg.name: asdict(cfg) for cfg in agile_candidates()},
        "datasets": datasets,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
