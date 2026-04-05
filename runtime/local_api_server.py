#!/usr/bin/env python3
"""
Local API and dashboard server.

Serves:
- built dashboard static assets from `dashboard/dist`
- JSON cache endpoints for runtime artifacts

This file intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_ROOT = BASE_DIR / "dashboard" / "dist"
DEFAULT_DASHBOARD_DATA_ROOT = BASE_DIR / "dashboard" / "public" / "data"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"


ARTIFACTS = {
    "dashboard": "../dashboard/public/data/dashboard.json",
    "live_signal": "live_signal_latest.json",
    "execution_plan": "execution_plan_latest.json",
    "execution_plan_base": "execution_plan_base_latest.json",
    "overlay": "active_overlay_latest.json",
    "overlay_raw": "anthropic_overlay_latest.json",
    "validation": "validation_pack.json",
    "learning_state": "learning_state.json",
    "paper_trading": "paper_trading_latest.json",
    "paper_base": "paper_base_latest.json",
    "paper_comparison": "paper_comparison_latest.json",
    "paper_backfill": "paper_backfill_latest.json",
    "audit_runs": "audit_runs_latest.json",
    "daily_cycle": "daily_cycle_latest.json",
    "execution_submissions": "execution_submissions_latest.json",
    "reconciliation": "reconciliation_latest.json",
    "execution_confirmation": "execution_confirmation_latest.json",
}


def json_response(status: str, path: Path, data: Any = None, *, exists: bool = True) -> Dict[str, Any]:
    return {
        "status": status,
        "path": str(path),
        "exists": exists,
        "data": data,
    }


class LocalAPIHandler(BaseHTTPRequestHandler):
    server_version = "TraderLocalAPI/1.0"

    @property
    def dashboard_root(self) -> Path:
        return self.server.dashboard_root  # type: ignore[attr-defined]

    @property
    def cache_dir(self) -> Path:
        return self.server.cache_dir  # type: ignore[attr-defined]

    @property
    def dashboard_data_root(self) -> Path:
        return self.server.dashboard_data_root  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Keep the console quiet unless there is an error.
        return

    def _send_headers(self, status_code: int, content_type: str, content_length: int) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        # Only advertise a wildcard CORS policy when the server is bound to a
        # loopback interface. If the operator exposes the server on a real
        # network (``--host 0.0.0.0`` etc.) we intentionally fall back to the
        # browser same-origin default so that cached portfolio state, overlays
        # and execution plans cannot be read cross-origin.
        if getattr(self.server, "cors_allow_any", False):
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self._send_headers(status_code, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _read_json_file(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _serve_api_index(self) -> None:
        payload = {
            "ok": True,
            "artifacts": list(ARTIFACTS.keys()),
            "dashboard_root": str(self.dashboard_root),
            "cache_dir": str(self.cache_dir),
        }
        self._send_json(payload)

    def _serve_artifact(self, name: str) -> None:
        filename = ARTIFACTS[name]
        if name == "dashboard":
            path = self.dashboard_data_root / "dashboard.json"
        else:
            path = self.cache_dir / filename
        if path.exists():
            try:
                data = self._read_json_file(path)
            except Exception as exc:
                self._send_json(json_response("error", path, {"message": str(exc)}, exists=True), status_code=500)
                return
            if name == "dashboard":
                self._send_json(data)
                return
            self._send_json(json_response("ok", path, data, exists=True))
            return
        self._send_json(json_response("missing", path, None, exists=False))

    def _resolve_static_path(self, request_path: str) -> Optional[Path]:
        if request_path.startswith("/data/"):
            data_root = self.dashboard_data_root
            if not data_root.exists():
                return None
            rel = request_path.removeprefix("/data/")
            candidate = (data_root / rel).resolve()
            try:
                candidate.relative_to(data_root.resolve())
            except ValueError:
                return None
            return candidate

        root = self.dashboard_root
        if not root.exists():
            return None

        rel = request_path.lstrip("/")
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    def _serve_static(self) -> None:
        root = self.dashboard_root
        if not root.exists():
            message = (
                "Dashboard build not found. Run:\n"
                "cd dashboard && npm install && npm run build\n"
            )
            body = message.encode("utf-8")
            self._send_headers(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain; charset=utf-8", len(body))
            self.wfile.write(body)
            return

        parsed = urlparse(self.path)
        request_path = parsed.path
        if request_path == "/":
            request_path = "/index.html"

        candidate = self._resolve_static_path(request_path)
        if candidate and candidate.is_file():
            content_type, _ = mimetypes.guess_type(candidate.name)
            content_type = content_type or "application/octet-stream"
            body = candidate.read_bytes()
            self._send_headers(HTTPStatus.OK, content_type, len(body))
            self.wfile.write(body)
            return

        if request_path.startswith("/data/"):
            body = b"Requested dashboard data file was not found."
            self._send_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", len(body))
            self.wfile.write(body)
            return

        index_path = root / "index.html"
        if index_path.exists():
            body = index_path.read_bytes()
            self._send_headers(HTTPStatus.OK, "text/html; charset=utf-8", len(body))
            self.wfile.write(body)
            return

        body = b"Dashboard build is present but index.html is missing."
        self._send_headers(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", len(body))
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/api":
            return self._serve_api_index()
        if path.startswith("/api/"):
            name = path.removeprefix("/api/").split(".")[0]
            if name in ARTIFACTS:
                return self._serve_artifact(name)
            payload = {"ok": False, "error": "unknown artifact", "known": list(ARTIFACTS.keys())}
            return self._send_json(payload, status_code=404)

        return self._serve_static()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve dashboard static files and cache JSON endpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dashboard-root", default=str(DEFAULT_DASHBOARD_ROOT))
    parser.add_argument("--dashboard-data-root", default=str(DEFAULT_DASHBOARD_DATA_ROOT))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    dashboard_root = Path(args.dashboard_root).expanduser().resolve()
    dashboard_data_root = Path(args.dashboard_data_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    server = ThreadingHTTPServer((args.host, args.port), LocalAPIHandler)
    server.dashboard_root = dashboard_root  # type: ignore[attr-defined]
    server.dashboard_data_root = dashboard_data_root  # type: ignore[attr-defined]
    server.cache_dir = cache_dir  # type: ignore[attr-defined]
    server.cors_allow_any = args.host in {"127.0.0.1", "localhost", "::1"}  # type: ignore[attr-defined]

    print(f"Serving dashboard from: {dashboard_root}")
    print(f"Serving dashboard data from: {dashboard_data_root}")
    print(f"Serving cache from: {cache_dir}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print(
        "API endpoints: /api, /api/dashboard, /api/live_signal, /api/execution_plan, /api/execution_plan_base, /api/overlay, /api/overlay_raw, "
        "/api/validation, /api/learning_state, /api/paper_trading, /api/paper_base, /api/paper_comparison, /api/paper_backfill, "
        "/api/audit_runs, /api/daily_cycle, /api/execution_submissions, /api/reconciliation, /api/execution_confirmation"
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
