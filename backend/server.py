#!/usr/bin/env python3
"""Local-only server for the public three-model explorer.

* Binds to 127.0.0.1 only (not configurable) and rejects non-loopback Host headers.
* Loads pinned checkpoints from the local Hugging Face cache on demand; never downloads.
* Never logs prompts: the request log records method, path (without query) and status.
* Keeps only the latest comparison in memory; writes nothing to disk.
* Sends a Content-Security-Policy that blocks the page from contacting any other origin,
  so a prompt can only ever reach this loopback backend -- never telemetry or a remote API.

Serving the catalog and the per-model architecture pages needs no model and no network;
only the live prompt comparison loads weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
UI_DIR = HERE.parent / "ui"
sys.path.insert(0, str(HERE))

import catalog          # noqa: E402
import explanations     # noqa: E402
import registry3        # noqa: E402
import compare3         # noqa: E402

HOST = "127.0.0.1"
MAX_BODY = 64 * 1024
# connect-src 'self' keeps every request same-origin (this loopback). No external origin
# is reachable from the page, so a prompt cannot be sent to telemetry or a remote model.
CSP = ("default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; "
       "script-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
       "form-action 'none'; frame-ancestors 'none'")

STORE = compare3.Compare3Store()


class Handler(SimpleHTTPRequestHandler):
    server_version = "ModelExplorer/0.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIR), **kw)

    def log_message(self, fmt, *args):  # never log bodies/queries
        pass

    def log_request(self, code="-", size="-"):
        path = urlsplit(self.path).path
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {path} {code}\n")

    def end_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._host_ok():
            return self._json(403, {"error": "loopback host required"})
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/health":
                return self._json(200, {"ok": True, "runnable_slots": registry3.available_slots()})
            if url.path == "/api/catalog":
                return self._json(200, catalog.catalog_payload())
            if url.path == "/api/metrics":
                return self._json(200, explanations.metrics_payload())
            if url.path == "/api/slots":
                return self._json(200, {"slots": registry3.available_slots()})
            if url.path == "/api/model":
                return self._json(200, catalog.public_view(catalog.by_id(q["id"])))
            if url.path == "/api/compare3/status":
                return self._json(200, STORE.status(q["job_id"]))
        except KeyError as e:
            return self._json(404, {"error": str(e)})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok():
            return self._json(403, {"error": "loopback host required"})
        path = urlsplit(self.path).path
        if path != "/api/compare3":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(413, {"error": f"body must be 1..{MAX_BODY} bytes"})
        try:
            req = json.loads(self.rfile.read(n))
            job = STORE.run_job(req)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        except KeyError as e:
            return self._json(400, {"error": f"unknown slot: {e}"})
        except RuntimeError as e:
            return self._json(409, {"error": str(e)})
        return self._json(202, {"job_id": job.id, "kind": job.kind})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    ready = registry3.available_slots()
    n_ready = sum(1 for v in ready.values() if v.get("ready"))
    sys.stderr.write(f"public explorer: {n_ready}/{len(ready)} runnable models verified in local cache\n")
    srv = ThreadingHTTPServer((HOST, args.port), Handler)
    sys.stderr.write(f"ready: http://{HOST}:{args.port}/  (Ctrl-C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
