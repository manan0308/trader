from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime import daily_cycle


def test_resolve_runtime_data_source_prefers_groww_when_ready(monkeypatch) -> None:
    monkeypatch.setattr(daily_cycle, "groww_auth_ready", lambda groww_universe_file=None, purpose="signal": True)
    assert daily_cycle.resolve_runtime_data_source("auto", purpose="signal") == "groww"


def test_resolve_runtime_data_source_falls_back_to_yfinance(monkeypatch) -> None:
    monkeypatch.setattr(daily_cycle, "groww_auth_ready", lambda groww_universe_file=None, purpose="signal": False)
    assert daily_cycle.resolve_runtime_data_source("auto", purpose="signal") == "yfinance"


def test_groww_auth_ready_uses_fresh_status_without_session(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "groww_auth_status.json"
    status_path.write_text(
        '{"status":"ok","assumed_token_expiry_ist":"2099-01-01T06:00:00+05:30","smoke_test_ok":true,"historical_candles_ok":true}',
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_cycle, "GROWW_AUTH_STATUS_PATH", status_path)
    monkeypatch.setattr(
        daily_cycle,
        "build_groww_session",
        lambda groww_universe_file=None: (_ for _ in ()).throw(AssertionError("session should not be built")),
    )
    assert daily_cycle.groww_auth_ready(purpose="signal") is True


def test_groww_auth_ready_requires_historical_for_signal(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "groww_auth_status.json"
    status_path.write_text(
        '{"status":"ok","assumed_token_expiry_ist":"2099-01-01T06:00:00+05:30","smoke_test_ok":true,"historical_candles_ok":false}',
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_cycle, "GROWW_AUTH_STATUS_PATH", status_path)
    assert daily_cycle.groww_auth_ready(purpose="signal") is False
    assert daily_cycle.groww_auth_ready(purpose="execution") is True
