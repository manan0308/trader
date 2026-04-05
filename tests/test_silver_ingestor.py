from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader_system.data.ingest_india_silver import choose_daily_close


def test_choose_daily_close_prefers_higher_volume_then_nearer_expiry() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-01", "close": 90000, "volume": 100, "open_interest": 1000, "contract": "SILVERFEB", "expiry": "2026-02-05", "source": "mcx", "source_file": "a"},
            {"date": "2026-01-01", "close": 90100, "volume": 200, "open_interest": 800, "contract": "SILVERJAN", "expiry": "2026-01-05", "source": "mcx", "source_file": "a"},
            {"date": "2026-01-02", "close": 90500, "volume": 150, "open_interest": 900, "contract": "SILVERFEB", "expiry": "2026-02-05", "source": "mcx", "source_file": "a"},
        ]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])

    chosen = choose_daily_close(frame)

    assert len(chosen) == 2
    assert chosen.iloc[0]["contract"] == "SILVERJAN"
    assert chosen.iloc[0]["close"] == 90100
