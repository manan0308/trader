#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from trader_system.runtime.store import (
    AUDIT_RUNS_PATH,
    DAILY_RUN_PATH,
    EFFECTIVE_OVERLAY_PATH,
    LEARNING_STATE_PATH,
    PAPER_SUMMARY_PATH,
    RAW_OVERLAY_PATH,
    read_json,
    read_jsonl,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = BASE_DIR / "dashboard"
STATIC_DIR = DASHBOARD_DIR / "dist" if (DASHBOARD_DIR / "dist").exists() else DASHBOARD_DIR / "public"


class RuntimeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api(parsed.path)
            return
        if parsed.path == "/":
            if (STATIC_DIR / "index.html").exists():
                self.path = "/index.html"
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    (
                        "Runtime server is up.\n"
                        "Use /api/health, /api/dashboard, /api/paper, /api/learning, /api/audit-runs.\n"
                        "For the React UI, run `npm run dev` in dashboard/ or build it first.\n"
                    ).encode("utf-8")
                )
                return
        return super().do_GET()

    def handle_api(self, path: str) -> None:
        payload = None
        if path == "/api/health":
            payload = {"ok": True}
        elif path == "/api/dashboard":
            payload = read_json(DASHBOARD_DIR / "public" / "data" / "dashboard.json", default={})
        elif path == "/api/paper":
            payload = read_json(PAPER_SUMMARY_PATH, default={})
        elif path == "/api/learning":
            payload = read_json(LEARNING_STATE_PATH, default={})
        elif path == "/api/audit-runs":
            payload = {"rows": read_jsonl(AUDIT_RUNS_PATH)[-50:]}
        elif path == "/api/overlay/raw":
            payload = read_json(RAW_OVERLAY_PATH, default={})
        elif path == "/api/overlay/effective":
            payload = read_json(EFFECTIVE_OVERLAY_PATH, default={})
        elif path == "/api/daily-run":
            payload = read_json(DAILY_RUN_PATH, default={})
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route")
            return

        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve runtime JSON endpoints and dashboard static files.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RuntimeHandler)
    print(f"Serving runtime app on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
