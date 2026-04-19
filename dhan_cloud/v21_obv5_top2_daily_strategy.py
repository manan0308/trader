#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime

from common import fetch_universe_history, format_weights_pct, write_payload_if_requested
from research.alpha_v21_price_volume_rotation import build_obv5_scores, build_top_n_targets, apply_rebalance_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Dhan Cloud wrapper for the highest-CAGR v21 challenger.")
    parser.add_argument("--lookback-days", type=int, default=500)
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD override for local testing")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    print("Starting Dhan Cloud v21 OBV top-2 daily strategy execution...")

    try:
        prices, volume, metadata = fetch_universe_history(
            lookback_calendar_days=args.lookback_days,
            end_date=args.end_date,
        )
        score_frame = build_obv5_scores(prices, volume)
        targets = build_top_n_targets(score_frame, top_n=2)
        weights = apply_rebalance_policy(targets, frequency="DAILY", trade_band=0.0, trade_step=1.0)
        latest = weights.iloc[-1]
        latest_scores = score_frame.iloc[-1].sort_values(ascending=False)

        current_time = datetime.now()
        strategy_data = {
            "timestamp": current_time.isoformat(),
            "status": "success",
            "message": "v21 OBV 5-day top-2 daily target generated from Dhan daily candles",
            "execution_id": f"exec_{int(time.time())}",
            "model_name": "v21_obv5_top2_daily",
            "latest_market_date": metadata.get("latest_market_date"),
            "data_source": metadata.get("data_source"),
            "requested_window": {
                "start": metadata.get("requested_start_date"),
                "end": metadata.get("requested_end_date"),
                "rows": metadata.get("rows"),
            },
            "target_weights_pct": format_weights_pct(latest),
            "top_score_assets": {
                asset: round(float(value), 4)
                for asset, value in latest_scores.head(3).items()
            },
        }

        print(f"Result: {json.dumps(strategy_data, indent=2)}")
        write_payload_if_requested(strategy_data, args.output_json)

    except Exception as e:
        print(f"Error in strategy execution: {str(e)}")
        raise

    print("Dhan Cloud v21 OBV top-2 daily strategy completed successfully")


if __name__ == "__main__":
    main()
