#!/usr/bin/env python3
"""
Iterative search runner for friend-style v22 models.

Purpose:
- keep sampling v22 candidates
- score them on public checkpoint fit + full-history sanity
- continuously write best candidates into cache/

This is the repo-side fallback for "automation" when the app automation tool
is not available in the current session.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.alpha_v18_agile_rotation import BROAD_SYMBOL_MAP, ETF_SYMBOL_MAP, load_parquet_prices
from strategy.v9_engine import ALL, schedule_flags


DEFAULT_BROAD = Path("/Users/mananagarwal/Desktop/historical data/dhan/all_daily_fy14_fy26_aligned.parquet")
DEFAULT_ETF = Path("/Users/mananagarwal/Desktop/historical data/dhan/etfs/all_daily_fy24_fy26_aligned.parquet")
DEFAULT_RECENT = Path("/Users/mananagarwal/Desktop/2nd brain/plant to image/trader/cache/dhan_recent_2026_03_18_2026_04_18.csv")
DEFAULT_BEST = Path("/Users/mananagarwal/Desktop/2nd brain/plant to image/trader/cache/alpha_v22_iterative_best.json")
DEFAULT_LOG = Path("/Users/mananagarwal/Desktop/2nd brain/plant to image/trader/cache/alpha_v22_iterative_history.jsonl")

RISKY = ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US"]
PRE_ALLOWED = ["NIFTY", "GOLD", "SILVER", "US"]

CHECKPOINTS = {
    "2026-03-13": {"ret": 0.0790, "weights": {"GOLD": 0.00, "US": 0.00, "CASH": 1.00}},
    "2026-03-24": {"ret": 0.0588, "weights": {"GOLD": 0.025, "US": 0.00, "CASH": 0.975}},
    "2026-04-02": {"ret": 0.0690, "weights": {"GOLD": 0.005, "US": 0.32, "CASH": 0.675}},
    "2026-04-06": {"ret": None, "weights": {"MIDCAP": 0.015, "NIFTY": 0.015, "GOLD": 0.08, "US": 0.39, "CASH": 0.50}},
    "2026-04-10": {"ret": 0.1020, "weights": {"GOLD": 0.11, "US": 0.79, "CASH": 0.10}},
    "2026-04-17": {"ret": 0.1300, "weights": {"GOLD": 0.22, "US": 0.65, "CASH": 0.13}},
}


@dataclass(frozen=True)
class CandidateConfig:
    frequency: str
    top_n: int
    temp: float
    step: float
    band: float
    w5: float
    w10: float
    w20: float
    w63: float
    pre_pos20_min: float
    pre_pos63_min: float
    post_pos20_min: float
    post_pos63_min: float
    invest1: float
    invest2: float
    invest3: float
    thr1: float
    thr2: float
    thr3: float
    max_asset: float
    us_cap: float
    us_bias: float
    gold_bias: float
    silver_bias: float
    gold_shift: float
    gold_m20_min: float
    scout_size: float
    scout_m5_min: float
    scout_m20_min: float
    nifty_floor: float


def candidate_name(cfg: CandidateConfig) -> str:
    return (
        f"v22_{cfg.frequency.lower()}_top{cfg.top_n}"
        f"_u{int(round(cfg.us_cap * 100))}"
        f"_g{int(round(cfg.gold_shift * 100))}"
        f"_mx{int(round(cfg.max_asset * 100))}"
        f"_sc{int(round(cfg.scout_size * 1000))}"
    )


def load_etf_prices(parquet_path: Path, recent_csv: Path | None) -> pd.DataFrame:
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
        prices = (
            pd.concat([prices, recent])
            .sort_index()
            .groupby(level=0)
            .last()
            .dropna()
        )
    return prices


def nearest_on_or_before(index: pd.DatetimeIndex, date_str: str) -> pd.Timestamp:
    target = pd.Timestamp(date_str)
    eligible = index[index <= target]
    if len(eligible) == 0:
        raise ValueError(date_str)
    return eligible[-1]


def softmax(values: pd.Series, temp: float) -> pd.Series:
    arr = values.astype(float).to_numpy()
    arr = arr - np.nanmax(arr)
    ex = np.exp(arr / max(temp, 1e-9))
    denom = ex.sum()
    out = ex / denom if denom > 0 else np.ones_like(arr) / len(arr)
    return pd.Series(out, index=values.index, dtype=float)


def build_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    risky = prices[RISKY]
    return {
        "m5": risky.pct_change(5),
        "m10": risky.pct_change(10),
        "m20": risky.pct_change(20),
        "m63": risky.pct_change(63),
        "returns": prices.pct_change(fill_method=None).fillna(0.0),
    }


def sample_candidate(rng: random.Random) -> CandidateConfig:
    weights = np.array([rng.uniform(0.0, 0.25), rng.uniform(0.0, 0.35), rng.uniform(0.2, 0.7), rng.uniform(0.2, 0.7)])
    weights = weights / weights.sum()
    return CandidateConfig(
        frequency=rng.choice(["DAILY", "WEEKLY"]),
        top_n=rng.choice([1, 2]),
        temp=rng.choice([0.04, 0.05, 0.07, 0.10]),
        step=rng.choice([0.50, 0.75, 1.00]),
        band=rng.choice([0.0, 0.01, 0.02, 0.05]),
        w5=float(weights[0]),
        w10=float(weights[1]),
        w20=float(weights[2]),
        w63=float(weights[3]),
        pre_pos20_min=rng.choice([-0.10, -0.05, -0.02, 0.0]),
        pre_pos63_min=rng.choice([-0.10, -0.05, -0.02, 0.0]),
        post_pos20_min=rng.choice([-0.10, -0.05, -0.02, 0.0]),
        post_pos63_min=rng.choice([-0.10, -0.05, -0.02, 0.0]),
        invest1=rng.choice([0.05, 0.10, 0.15]),
        invest2=rng.choice([0.25, 0.32, 0.35, 0.40, 0.50]),
        invest3=rng.choice([0.75, 0.80, 0.85, 0.90]),
        thr1=rng.choice([0.00, 0.01, 0.02]),
        thr2=rng.choice([0.03, 0.04, 0.05, 0.06]),
        thr3=rng.choice([0.08, 0.10, 0.12, 0.15]),
        max_asset=rng.choice([0.50, 0.60, 0.70, 0.80]),
        us_cap=rng.choice([0.65, 0.70, 0.75, 0.79, 0.80]),
        us_bias=rng.choice([0.00, 0.01, 0.02, 0.04]),
        gold_bias=rng.choice([0.00, 0.01, 0.02, 0.04]),
        silver_bias=rng.choice([0.00, 0.01, 0.02, 0.04]),
        gold_shift=rng.choice([0.00, 0.05, 0.08, 0.10, 0.12]),
        gold_m20_min=rng.choice([-0.05, -0.02, 0.0]),
        scout_size=rng.choice([0.0, 0.01, 0.015, 0.02, 0.03]),
        scout_m5_min=rng.choice([-0.02, 0.0, 0.01]),
        scout_m20_min=rng.choice([-0.10, -0.08, -0.05, 0.0]),
        nifty_floor=rng.choice([0.0, 0.005, 0.01, 0.02]),
    )


def build_weights(prices: pd.DataFrame, feats: Dict[str, pd.DataFrame], cfg: CandidateConfig) -> pd.DataFrame:
    m5 = feats["m5"]
    m10 = feats["m10"]
    m20 = feats["m20"]
    m63 = feats["m63"]

    if cfg.frequency == "DAILY":
        schedule = pd.Series(True, index=prices.index)
    else:
        schedule = schedule_flags(prices.index, "WEEKLY")

    weights = pd.DataFrame(0.0, index=prices.index, columns=ALL)
    current = pd.Series(0.0, index=ALL)
    current["CASH"] = 1.0

    for dt in prices.index:
        if pd.isna(m63.loc[dt]).all():
            weights.loc[dt] = current
            continue

        if not bool(schedule.loc[dt]):
            weights.loc[dt] = current
            continue

        allowed = PRE_ALLOWED if dt < pd.Timestamp("2026-04-01") else RISKY

        score = (
            cfg.w5 * m5.loc[dt].fillna(-999.0)
            + cfg.w10 * m10.loc[dt].fillna(-999.0)
            + cfg.w20 * m20.loc[dt].fillna(-999.0)
            + cfg.w63 * m63.loc[dt].fillna(-999.0)
        )

        score["US"] += cfg.us_bias
        score["GOLD"] += cfg.gold_bias
        score["SILVER"] += cfg.silver_bias

        eligible = pd.Series(False, index=RISKY)
        eligible.loc[allowed] = True

        if dt < pd.Timestamp("2026-04-01"):
            eligible &= m20.loc[dt].fillna(-999.0) > cfg.pre_pos20_min
            eligible &= m63.loc[dt].fillna(-999.0) > cfg.pre_pos63_min
        else:
            eligible &= m20.loc[dt].fillna(-999.0) > cfg.post_pos20_min
            eligible &= m63.loc[dt].fillna(-999.0) > cfg.post_pos63_min

        score[~eligible] = -999.0
        valid = score[score > -900.0].sort_values(ascending=False)

        invest = 0.0
        if len(valid):
            top = float(valid.iloc[0])
            if top > cfg.thr1:
                invest = cfg.invest1
            if top > cfg.thr2:
                invest = cfg.invest2
            if top > cfg.thr3:
                invest = cfg.invest3

        proposal = pd.Series(0.0, index=ALL)
        if invest > 0 and len(valid):
            picks = valid.index[: cfg.top_n]
            probs = softmax(valid.loc[picks], cfg.temp)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            proposal.loc[picks] = probs * invest

            if proposal["US"] > 0 and float(m20.loc[dt].get("GOLD", -1.0)) > cfg.gold_m20_min:
                shift = min(cfg.gold_shift, float(proposal["US"]))
                proposal["US"] -= shift
                proposal["GOLD"] += shift

            if dt >= pd.Timestamp("2026-04-01") and cfg.scout_size > 0.0:
                if float(m5.loc[dt].get("MIDCAP", -1.0)) > cfg.scout_m5_min and float(m20.loc[dt].get("MIDCAP", -1.0)) > cfg.scout_m20_min:
                    shift = min(cfg.scout_size, max(0.0, 1.0 - float(proposal.sum())))
                    proposal["MIDCAP"] += shift
                elif float(m5.loc[dt].get("NIFTY", -1.0)) > cfg.scout_m5_min and float(m20.loc[dt].get("NIFTY", -1.0)) > cfg.scout_m20_min:
                    shift = min(cfg.scout_size, max(0.0, 1.0 - float(proposal.sum())))
                    proposal["NIFTY"] += shift

            if proposal["US"] > cfg.us_cap:
                excess = float(proposal["US"] - cfg.us_cap)
                proposal["US"] = cfg.us_cap
                proposal["GOLD"] += excess

            # Keep a tiny NIFTY presence alive in the model family if room exists.
            if proposal["NIFTY"] < cfg.nifty_floor and dt < pd.Timestamp("2026-04-01"):
                room = max(0.0, 1.0 - float(proposal.sum()))
                shift = min(cfg.nifty_floor - float(proposal["NIFTY"]), room)
                if shift > 0:
                    proposal["NIFTY"] += shift

            if float(proposal.sum()) > 1.0:
                proposal = proposal / float(proposal.sum())

            # single-asset cap after all adjustments
            for _ in range(8):
                over = proposal[RISKY] > cfg.max_asset
                if not bool(over.any()):
                    break
                excess = float((proposal[RISKY][over] - cfg.max_asset).sum())
                capped = proposal[RISKY].copy()
                capped[over] = cfg.max_asset
                under = (capped > 0.0) & (~over)
                under_sum = float(capped[under].sum())
                if excess > 0 and under_sum > 0:
                    capped[under] += capped[under] / under_sum * excess
                proposal.loc[RISKY] = capped
                total = float(proposal[RISKY].sum())
                if total > 1.0:
                    proposal.loc[RISKY] = proposal[RISKY] / total

        proposal["CASH"] = max(0.0, 1.0 - float(proposal.drop(labels=["CASH"], errors="ignore").sum()))
        total = float(proposal.sum())
        if total > 0:
            proposal = proposal / total
        else:
            proposal["CASH"] = 1.0

        next_weights = current * (1.0 - cfg.step) + proposal * cfg.step
        next_weights = next_weights.clip(lower=0.0)
        next_weights = next_weights / float(next_weights.sum())
        if float((next_weights - current).abs().max()) > cfg.band:
            current = next_weights
        weights.loc[dt] = current

    return weights


def quick_metrics(prices: pd.DataFrame, weights: pd.DataFrame, *, start: str) -> Dict[str, object]:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    aligned = weights.reindex(prices.index).ffill().fillna(0.0)
    lagged = aligned.shift(1).bfill().fillna(0.0)
    turnover = aligned.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    net = ((lagged * returns).sum(axis=1) - turnover * 0.003).iloc[1:]
    window = net.loc[start:]
    equity = (1.0 + window).cumprod()
    years = max(len(window) / 252.0, 1 / 252.0)
    vol = float(window.std() * math.sqrt(252.0)) if len(window) else 0.0
    sharpe = float(window.mean() * 252.0 / vol) if vol > 0 else 0.0
    mdd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if len(equity) else 0.0
    avg_cash = float(aligned.loc[window.index, "CASH"].mean()) if len(window) else 1.0
    return {
        "net": window,
        "equity": equity,
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_cash": avg_cash,
        "turnover_annualized": float((turnover.loc[window.index]).sum() / years) if len(window) else 0.0,
    }


def evaluate_candidate(
    broad_prices: pd.DataFrame,
    broad_feats: Dict[str, pd.DataFrame],
    etf_prices: pd.DataFrame,
    etf_feats: Dict[str, pd.DataFrame],
    cfg: CandidateConfig,
) -> Dict[str, object]:
    broad_weights = build_weights(broad_prices, broad_feats, cfg)
    etf_weights = build_weights(etf_prices, etf_feats, cfg)

    broad = quick_metrics(broad_prices, broad_weights, start="2014-01-01")
    etf = quick_metrics(etf_prices, etf_weights, start=str(etf_prices.index.min().date()))

    checkpoint_loss = 0.0
    checkpoint_rows: Dict[str, object] = {}
    broad_eq = broad["equity"]
    for ds, target in CHECKPOINTS.items():
        dt = nearest_on_or_before(etf_weights.index, ds)
        row = {
            "date_used": str(dt.date()),
            "ret_since_dec10": float(broad_eq.loc[dt] - 1.0) if dt in broad_eq.index else None,
            "weights": {asset: float(etf_weights.loc[dt, asset]) for asset in ["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]},
        }
        checkpoint_rows[ds] = row
        if target["ret"] is not None and row["ret_since_dec10"] is not None:
            checkpoint_loss += 15.0 * abs(float(row["ret_since_dec10"]) - float(target["ret"]))
        for asset, wanted in target["weights"].items():
            checkpoint_loss += abs(float(row["weights"].get(asset, 0.0)) - float(wanted))

    # One-year windows that the user asked about.
    broad_aligned = broad_weights.reindex(broad_prices.index).ffill().fillna(0.0)
    broad_returns = broad_prices.pct_change(fill_method=None).fillna(0.0)
    broad_lag = broad_aligned.shift(1).bfill().fillna(0.0)
    broad_turn = broad_aligned.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    broad_net = ((broad_lag * broad_returns).sum(axis=1) - broad_turn * 0.003).iloc[1:]

    def window_return(start: str, end: str) -> float:
        s = nearest_on_or_before(broad_net.index, start)
        e = nearest_on_or_before(broad_net.index, end)
        window = broad_net.loc[s:e]
        return float((1.0 + window).prod() - 1.0)

    ret_dec = window_return("2024-12-01", "2025-12-31")
    ret_jan = window_return("2025-01-01", "2026-01-31")

    # Multi-objective fit:
    # 1) public checkpoint fit
    # 2) not awful on broad history
    # 3) yearly windows near the claimed rough 30-40% range
    # 4) turnover around the public claim
    loss = checkpoint_loss
    loss += 6.0 * abs(ret_dec - 0.40)
    loss += 4.0 * abs(ret_jan - 0.35)
    loss += 3.0 * max(0.0, 0.10 - float(broad["cagr"]))
    loss += 6.0 * max(0.0, 0.30 - float(broad["sharpe"]))
    loss += 3.0 * max(0.0, float(broad["avg_cash"]) - 0.60)
    loss += 2.0 * max(0.0, float(etf["avg_cash"]) - 0.55)
    loss += 0.75 * abs(float(etf["turnover_annualized"]) - 3.76)
    loss += 2.0 * max(0.0, 0.15 + float(broad["mdd"]))

    return {
        "name": candidate_name(cfg),
        "loss": float(loss),
        "checkpoint_fit": float(checkpoint_loss),
        "config": asdict(cfg),
        "broad_cagr": float(broad["cagr"]),
        "broad_sharpe": float(broad["sharpe"]),
        "broad_max_drawdown": float(broad["mdd"]),
        "broad_avg_cash": float(broad["avg_cash"]),
        "etf_cagr": float(etf["cagr"]),
        "etf_sharpe": float(etf["sharpe"]),
        "etf_max_drawdown": float(etf["mdd"]),
        "etf_avg_cash": float(etf["avg_cash"]),
        "etf_turnover_annualized": float(etf["turnover_annualized"]),
        "one_year_dec24_dec25": ret_dec,
        "one_year_jan25_jan26": ret_jan,
        "checkpoints": checkpoint_rows,
    }


def save_best(best_rows: List[Dict[str, object]], best_path: Path, history_path: Path, cycle: int, samples_done: int) -> None:
    payload = {
        "updated_at_epoch": time.time(),
        "cycle": cycle,
        "samples_done": samples_done,
        "best": best_rows,
    }
    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative v22 search runner.")
    parser.add_argument("--broad-parquet", default=str(DEFAULT_BROAD))
    parser.add_argument("--etf-parquet", default=str(DEFAULT_ETF))
    parser.add_argument("--recent-csv", default=str(DEFAULT_RECENT))
    parser.add_argument("--best-json", default=str(DEFAULT_BEST))
    parser.add_argument("--history-jsonl", default=str(DEFAULT_LOG))
    parser.add_argument("--samples-per-cycle", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    broad_prices = load_parquet_prices(Path(args.broad_parquet).expanduser().resolve(), BROAD_SYMBOL_MAP)
    etf_prices = load_etf_prices(Path(args.etf_parquet).expanduser().resolve(), Path(args.recent_csv).expanduser().resolve())
    broad_feats = build_features(broad_prices)
    etf_feats = build_features(etf_prices)

    best_rows: List[Dict[str, object]] = []
    cycle = 0
    samples_done = 0

    while True:
        cycle += 1
        rows = list(best_rows)
        for _ in range(args.samples_per_cycle):
            cfg = sample_candidate(rng)
            row = evaluate_candidate(broad_prices, broad_feats, etf_prices, etf_feats, cfg)
            rows.append(row)
            samples_done += 1
        rows.sort(key=lambda item: float(item["loss"]))
        best_rows = rows[: args.top_k]
        save_best(best_rows, Path(args.best_json).expanduser().resolve(), Path(args.history_jsonl).expanduser().resolve(), cycle, samples_done)
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "samples_done": samples_done,
                    "best_name": best_rows[0]["name"] if best_rows else None,
                    "best_loss": best_rows[0]["loss"] if best_rows else None,
                    "best_checkpoint_fit": best_rows[0]["checkpoint_fit"] if best_rows else None,
                    "best_one_year_dec24_dec25": best_rows[0]["one_year_dec24_dec25"] if best_rows else None,
                    "best_broad_cagr": best_rows[0]["broad_cagr"] if best_rows else None,
                    "best_etf_cagr": best_rows[0]["etf_cagr"] if best_rows else None,
                }
            ),
            flush=True,
        )
        if args.cycles > 0 and cycle >= args.cycles:
            break
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
