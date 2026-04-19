#!/usr/bin/env python3
"""
ALPHA ENGINE v23 - JPM MOMENTUM PLAYBOOK
========================================

Paper-inspired research variants adapted from:
  "Momentum Strategies Across Asset Classes" (J.P. Morgan, April 2015)

What we port into our small long-only ETF universe:
- Long-only absolute momentum with longer lookbacks (6-12 months)
- Hybrid long-only momentum: require both relative and absolute momentum
- Risk-adjusted momentum ranking (return / volatility)
- Multi-signal trend scorecard
- Diversified trend-following through blending simple signals
- A stop-loss / block-out variant

What we explicitly do NOT port one-for-one:
- Short books (user wants long only / live deployable ideas)
- Leverage beyond 100% gross
- The paper's full global futures universe

The goal is not to claim fidelity to the bank index products. The goal is to
see which ideas survive once adapted to our India-centric multi-asset book.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    ETF_SYMBOL_MAP,
    cap_and_redistribute,
    fixed_config_wfo,
    load_parquet_prices,
    rotation_summary,
    strip_metrics,
)
from strategy.v9_engine import (
    ALL,
    CACHE_DIR,
    DEFAULT_RF,
    DEFAULT_TX,
    RISKY,
    build_features,
    inverse_vol_weights,
    performance_metrics,
    print_results_table,
    run_strategy,
)


DEFAULT_RECENT = Path("/Users/mananagarwal/Desktop/2nd brain/plant to image/trader/cache/dhan_recent_2026_03_18_2026_04_18.csv")
DEFAULT_OUTPUT = CACHE_DIR / "alpha_v23_jpm_momentum_playbook_results.json"


@dataclass(frozen=True)
class JPMVariant:
    name: str
    family: str
    hold_days: int = 21
    top_n: int = 3
    max_asset_weight: float = 0.45
    min_cash: float = 0.0
    stop_loss: float = 0.0
    block_days: int = 0


def paper_variants() -> List[JPMVariant]:
    return [
        JPMVariant(
            name="v23_jpm_abs12_allpos",
            family="abs12_allpos",
            hold_days=21,
            top_n=6,
            max_asset_weight=0.35,
            min_cash=0.0,
        ),
        JPMVariant(
            name="v23_jpm_hybrid6_top3",
            family="hybrid6_top3",
            hold_days=21,
            top_n=3,
            max_asset_weight=0.45,
            min_cash=0.05,
        ),
        JPMVariant(
            name="v23_jpm_hybrid6_retvol_top3",
            family="hybrid6_retvol_top3",
            hold_days=21,
            top_n=3,
            max_asset_weight=0.45,
            min_cash=0.05,
        ),
        JPMVariant(
            name="v23_jpm_scorecard8_top3",
            family="scorecard8_top3",
            hold_days=21,
            top_n=3,
            max_asset_weight=0.45,
            min_cash=0.05,
        ),
        JPMVariant(
            name="v23_jpm_dtf3signal_top3",
            family="dtf3signal_top3",
            hold_days=21,
            top_n=3,
            max_asset_weight=0.45,
            min_cash=0.05,
        ),
        JPMVariant(
            name="v23_jpm_hybrid6_stop5_block21",
            family="hybrid6_top3",
            hold_days=21,
            top_n=3,
            max_asset_weight=0.45,
            min_cash=0.10,
            stop_loss=0.05,
            block_days=21,
        ),
    ]


def load_etf_prices(parquet_path: Path, recent_csv: Optional[Path]) -> pd.DataFrame:
    prices = load_parquet_prices(parquet_path, ETF_SYMBOL_MAP)
    if recent_csv and recent_csv.exists():
        recent_raw = pd.read_csv(recent_csv, parse_dates=["date"])
        recent = recent_raw[recent_raw["symbol"].isin(ETF_SYMBOL_MAP)].copy()
        recent["asset"] = recent["symbol"].map(ETF_SYMBOL_MAP)
        recent = (
            recent.pivot(index="date", columns="asset", values="close")
            .sort_index()
            .reindex(columns=ALL)
        )
        prices = pd.concat([prices, recent]).sort_index().groupby(level=0).last().dropna()
    return prices


def build_paper_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    base = build_features(prices)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    risky = prices[RISKY]
    out: Dict[str, pd.DataFrame | pd.Series] = dict(base)
    out["close"] = risky
    out["ret21"] = risky.pct_change(21, fill_method=None)
    out["ret42"] = risky.pct_change(42, fill_method=None)
    out["ret63"] = risky.pct_change(63, fill_method=None)
    out["ret126"] = risky.pct_change(126, fill_method=None)
    out["ret252"] = risky.pct_change(252, fill_method=None)
    out["ret10"] = risky.pct_change(10, fill_method=None)
    out["sma5"] = risky.rolling(5).mean()
    out["sma10"] = risky.rolling(10).mean()
    out["sma100"] = risky.rolling(100).mean()
    out["vol126"] = returns[RISKY].rolling(126).std() * np.sqrt(252.0)
    out["vol252"] = returns[RISKY].rolling(252).std() * np.sqrt(252.0)
    return out


def row_rank(row: pd.Series) -> pd.Series:
    ranked = pd.Series(row, dtype=float).replace([np.inf, -np.inf], np.nan)
    if ranked.notna().sum() <= 1:
        return pd.Series(0.5, index=ranked.index, dtype=float)
    return ranked.rank(pct=True, method="average", ascending=True).fillna(0.5)


def selected_target(
    selected: List[str],
    vol_row: pd.Series,
    *,
    max_asset_weight: float,
    min_cash: float,
) -> pd.Series:
    target = pd.Series(0.0, index=ALL, dtype=float)
    if selected:
        inv = inverse_vol_weights(vol_row, selected)
        inv = cap_and_redistribute(inv, max_asset_weight)
        target.loc[selected] = (1.0 - min_cash) * inv
    target["CASH"] = max(0.0, 1.0 - float(target[RISKY].sum()))
    total = float(target.sum())
    return target / total if total > 0 else pd.Series([0, 0, 0, 0, 0, 0, 1], index=ALL, dtype=float)


def target_abs12_allpos(dt: pd.Timestamp, feats: Dict[str, pd.DataFrame | pd.Series], cfg: JPMVariant) -> pd.Series:
    ret252 = pd.DataFrame(feats["ret252"]).loc[dt]
    vol63 = pd.DataFrame(feats["vol63"]).loc[dt]
    eligible = ret252[ret252 > 0.0].sort_values(ascending=False)
    selected = list(eligible.index[: cfg.top_n])
    return selected_target(selected, vol63, max_asset_weight=cfg.max_asset_weight, min_cash=cfg.min_cash)


def target_hybrid6(dt: pd.Timestamp, feats: Dict[str, pd.DataFrame | pd.Series], cfg: JPMVariant, *, risk_adjust: bool) -> pd.Series:
    ret126 = pd.DataFrame(feats["ret126"]).loc[dt]
    vol126 = pd.DataFrame(feats["vol126"]).loc[dt]
    sma200 = pd.DataFrame(feats["sma200"]).loc[dt]
    close = pd.DataFrame(feats["close"]).loc[dt]
    eligible = ret126 > 0.0
    eligible &= close > sma200
    score = ret126 / vol126.replace(0.0, np.nan) if risk_adjust else ret126
    score = score.replace([np.inf, -np.inf], np.nan)
    ranked = score[eligible].sort_values(ascending=False)
    selected = list(ranked.index[: cfg.top_n])
    return selected_target(selected, pd.DataFrame(feats["vol63"]).loc[dt], max_asset_weight=cfg.max_asset_weight, min_cash=cfg.min_cash)


def target_scorecard8(dt: pd.Timestamp, feats: Dict[str, pd.DataFrame | pd.Series], cfg: JPMVariant) -> pd.Series:
    ret126 = pd.DataFrame(feats["ret126"]).loc[dt]
    ret252 = pd.DataFrame(feats["ret252"]).loc[dt]
    ret63 = pd.DataFrame(feats["ret63"]).loc[dt]
    ret10 = pd.DataFrame(feats["ret10"]).loc[dt]
    vol63 = pd.DataFrame(feats["vol63"]).loc[dt]
    vol252 = pd.DataFrame(feats["vol252"]).loc[dt]
    close = pd.DataFrame(feats["close"]).loc[dt]
    sma5 = pd.DataFrame(feats["sma5"]).loc[dt]
    sma10 = pd.DataFrame(feats["sma10"]).loc[dt]
    sma100 = pd.DataFrame(feats["sma100"]).loc[dt]
    sma200 = pd.DataFrame(feats["sma200"]).loc[dt]

    rank_inputs = pd.DataFrame(
        {
            "ret126": row_rank(ret126),
            "ret252": row_rank(ret252),
            "ret63_vol": row_rank(ret63 / vol63.replace(0.0, np.nan)),
            "ret252_vol": row_rank(ret252 / vol252.replace(0.0, np.nan)),
            "ma200": (close > sma200).astype(float),
            "xover5_100": (sma5 > sma100).astype(float),
            "xover10_200": (sma10 > sma200).astype(float),
            "ret126_minus_ret10": row_rank(ret126 - ret10),
        }
    )
    score = rank_inputs.mean(axis=1)
    eligible = (ret126 > 0.0) | (ret252 > 0.0)
    ranked = score[eligible].sort_values(ascending=False)
    selected = list(ranked.index[: cfg.top_n])
    return selected_target(selected, vol63, max_asset_weight=cfg.max_asset_weight, min_cash=cfg.min_cash)


def target_xover10_200(dt: pd.Timestamp, feats: Dict[str, pd.DataFrame | pd.Series], cfg: JPMVariant) -> pd.Series:
    sma10 = pd.DataFrame(feats["sma10"]).loc[dt]
    sma200 = pd.DataFrame(feats["sma200"]).loc[dt]
    ret126 = pd.DataFrame(feats["ret126"]).loc[dt]
    vol63 = pd.DataFrame(feats["vol63"]).loc[dt]
    eligible = (sma10 > sma200) & (ret126 > -0.05)
    score = ret126.where(eligible, np.nan)
    ranked = score.dropna().sort_values(ascending=False)
    selected = list(ranked.index[: cfg.top_n])
    return selected_target(selected, vol63, max_asset_weight=cfg.max_asset_weight, min_cash=cfg.min_cash)


def target_dtf3signal(dt: pd.Timestamp, feats: Dict[str, pd.DataFrame | pd.Series], cfg: JPMVariant) -> pd.Series:
    t1 = target_abs12_allpos(dt, feats, cfg)
    # Shorter signal inside the blend, but still long-only and filtered.
    ret21 = pd.DataFrame(feats["ret21"]).loc[dt]
    vol63 = pd.DataFrame(feats["vol63"]).loc[dt]
    selected21 = list(ret21[ret21 > 0.0].sort_values(ascending=False).index[: cfg.top_n])
    t2 = selected_target(selected21, vol63, max_asset_weight=cfg.max_asset_weight, min_cash=cfg.min_cash)
    t3 = target_xover10_200(dt, feats, cfg)
    blend = (t1 + t2 + t3) / 3.0
    blend["CASH"] = max(0.0, 1.0 - float(blend[RISKY].sum()))
    return blend / float(blend.sum())


def run_long_only_paper_variant(prices: pd.DataFrame, cfg: JPMVariant) -> pd.DataFrame:
    feats = build_paper_features(prices)
    returns = pd.DataFrame(feats["returns"]).reindex(columns=ALL).fillna(0.0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL, dtype=float)
    current = pd.Series(0.0, index=ALL, dtype=float)
    current["CASH"] = 1.0

    equity = 1.0
    peak = 1.0
    blocked_left = 0
    warmup = 260
    step = min(1.0, 1.0 / max(cfg.hold_days, 1))

    for i, dt in enumerate(prices.index):
        if i > 0:
            day_ret = float((current * returns.loc[dt]).sum())
            equity *= 1.0 + day_ret
            peak = max(peak, equity)
            if cfg.stop_loss > 0 and peak > 0 and blocked_left <= 0:
                if equity / peak - 1.0 <= -cfg.stop_loss:
                    blocked_left = cfg.block_days

        if i < warmup:
            weights.iloc[i] = current
            continue

        if blocked_left > 0:
            target = pd.Series(0.0, index=ALL, dtype=float)
            target["CASH"] = 1.0
            blocked_left -= 1
        else:
            if cfg.family == "abs12_allpos":
                target = target_abs12_allpos(dt, feats, cfg)
            elif cfg.family == "hybrid6_top3":
                target = target_hybrid6(dt, feats, cfg, risk_adjust=False)
            elif cfg.family == "hybrid6_retvol_top3":
                target = target_hybrid6(dt, feats, cfg, risk_adjust=True)
            elif cfg.family == "scorecard8_top3":
                target = target_scorecard8(dt, feats, cfg)
            elif cfg.family == "dtf3signal_top3":
                target = target_dtf3signal(dt, feats, cfg)
            else:
                raise ValueError(f"Unsupported family: {cfg.family}")

        proposal = current * (1.0 - step) + target * step
        proposal = proposal.clip(lower=0.0)
        proposal["CASH"] = max(0.0, 1.0 - float(proposal[RISKY].sum()))
        current = proposal / float(proposal.sum())
        weights.iloc[i] = current

    return weights


def run_dataset(
    name: str,
    prices: pd.DataFrame,
    rf: float,
    tx_cost: float,
    train_days: int,
    test_days: int,
    cost_model: Optional[IndianDeliveryCostModel],
    base_value: float,
) -> Dict[str, object]:
    runners: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "v9_quant": lambda frame: run_strategy(frame, BASE_V9, overlay=None),
    }
    for cfg in paper_variants():
        runners[cfg.name] = lambda frame, cfg=cfg: run_long_only_paper_variant(frame, cfg)

    full_sample: Dict[str, object] = {}
    full_rows: List[Dict[str, object]] = []
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
        full_sample[label] = {
            "metrics": strip_metrics(metrics),
            "rotation": rotation_summary(weights),
        }
        full_rows.append(metrics)
    print_results_table(f"FULL SAMPLE - {name}", full_rows)

    walk_forward: Dict[str, object] = {}
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
        walk_forward[label] = {
            "metrics": strip_metrics(payload["metrics"]),
            "windows": payload["windows"],
            "rotation": rotation_summary(payload["metrics"]["weights"]),
        }
        wfo_rows.append(payload["metrics"])
    print_results_table(f"WALK-FORWARD OOS - {name}", wfo_rows)

    return {
        "sample_start": prices.index[0].strftime("%Y-%m-%d"),
        "sample_end": prices.index[-1].strftime("%Y-%m-%d"),
        "rows": int(len(prices)),
        "full_sample": full_sample,
        "walk_forward": walk_forward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JPM-paper-inspired momentum variants on our broad and ETF datasets.")
    parser.add_argument("--broad-parquet", default=str(DEFAULT_DHAN))
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--recent-csv", default=str(DEFAULT_RECENT))
    parser.add_argument("--rf", type=float, default=DEFAULT_RF)
    parser.add_argument("--tx-bps", type=float, default=30.0)
    parser.add_argument("--cost-model", choices=["flat", "india_delivery"], default="india_delivery")
    parser.add_argument("--base-value", type=float, default=1_000_000.0)
    parser.add_argument("--train-days", type=int, default=756)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    broad_prices = load_parquet_prices(Path(args.broad_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    etf_prices = load_etf_prices(
        Path(args.etf_parquet).expanduser().resolve(),
        Path(args.recent_csv).expanduser().resolve() if args.recent_csv else None,
    )

    tx_cost = args.tx_bps / 10_000.0
    cost_model = resolve_cost_model(args.cost_model)

    broad_payload = run_dataset(
        "broad_dhan_fy14_fy26",
        broad_prices,
        rf=args.rf,
        tx_cost=tx_cost,
        train_days=args.train_days,
        test_days=args.test_days,
        cost_model=cost_model,
        base_value=args.base_value,
    )
    etf_payload = run_dataset(
        "etf_dhan_fy24_fy26_plus_recent",
        etf_prices,
        rf=args.rf,
        tx_cost=tx_cost,
        train_days=min(args.train_days, 504),
        test_days=min(args.test_days, 63),
        cost_model=cost_model,
        base_value=args.base_value,
    )

    payload = {
        "paper": {
            "title": "Momentum Strategies Across Asset Classes",
            "author": "J.P. Morgan Quantitative and Derivatives Strategy",
            "date": "2015-04-15",
            "actionable_takeaways": [
                "Longer 6-12 month momentum signals generally beat 1-3 month signals after costs.",
                "Long-only hybrid momentum can reduce drawdowns by requiring both relative and absolute confirmation.",
                "Risk-adjusted momentum (return divided by volatility) improves cross-asset ranking, especially in smaller baskets.",
                "Diversifying across signals often helps robustness more than hunting for one perfect indicator.",
                "Fast short-term momentum is fragile after realistic transaction costs.",
                "A modest stop-loss / block-out can help, but it is not a free lunch.",
            ],
        },
        "tx_cost_bps": args.tx_bps,
        "cost_model": args.cost_model,
        "base_value": args.base_value,
        "variants": [cfg.__dict__ for cfg in paper_variants()],
        "datasets": {
            "broad": broad_payload,
            "etf": etf_payload,
        },
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
