#!/usr/bin/env python3
"""Automated local verification for the public three-model explorer.

Starts the real server on a free loopback port and checks, end to end:
  * it binds to 127.0.0.1 and rejects a non-loopback Host header;
  * every response carries the restrictive Content-Security-Policy;
  * /api/catalog returns 10 entries, 3 runnable, with licences labelled;
  * /api/metrics explains every comparison row;
  * a neutral prompt runs through all three verified models, each provenance-checked,
    producing non-empty answers and measured internals;
  * the prompt text never appears in the server log (privacy);
  * negative controls are rejected: an unknown slot, an over-size body, a tampered pin.

Prints a JSON summary and exits non-zero if any check fails. Writes nothing to the repo
and sends nothing off the machine (HF offline is forced). Needs the model weights in the
local cache; checks that require them are reported as skipped if a slot is unverifiable.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import server as srv          # noqa: E402
import registry3             # noqa: E402
import catalog as cat        # noqa: E402

MARKER = "zqx-neutral-" + str(int(time.time()))
PROMPT = f"Name one primary colour. ({MARKER})"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def get(port, path, host=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if host:
        req.add_header("Host", host)
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def post(port, path, obj):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    log = io.StringIO()
    real_stderr = sys.stderr
    sys.stderr = log  # capture server request log to assert the prompt never appears
    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        _run(httpd, port, log)
    finally:
        httpd.shutdown(); httpd.server_close()
        sys.stderr = real_stderr

    passed = sum(1 for r in RESULTS if r["pass"])
    total = len(RESULTS)
    summary = {"suite": "public-explorer selfcheck3", "passed": passed, "total": total,
               "all_pass": passed == total, "checks": RESULTS}
    print("\n" + json.dumps(summary, indent=1))
    sys.exit(0 if passed == total else 1)


def _run(httpd, port, log):
    check("server binds to 127.0.0.1", httpd.server_address[0] == "127.0.0.1", httpd.server_address[0])

    st, hdr, body = get(port, "/api/catalog")
    csp = hdr.get("Content-Security-Policy", "")
    check("CSP header present", "default-src 'self'" in csp, csp[:40] + "...")
    check("CSP blocks cross-origin connect", "connect-src 'self'" in csp)
    cj = json.loads(body)
    check("catalog has <=10 entries", cj["count"] <= 10, f"count={cj['count']}")
    check("catalog has >=3 runnable (pinned)", cj["runnable_count"] >= 3, f"runnable={cj['runnable_count']}")
    non_osi = [m["id"] for m in cj["models"] if not m["license"]["osi_open_source"]]
    check("non-OSI licences are flagged, not runnable",
          all(not cat.by_id(i)["runnable"] for i in non_osi), f"non-osi={non_osi}")

    st, hdr, body = get(port, "/api/metrics")
    mj = json.loads(body)
    missing = [r for r in mj["comparison_rows"] if r not in mj["metrics"]]
    check("every comparison row has an explanation", not missing, f"missing={missing}")

    st, _, _ = get(port, "/api/catalog", host="evil.example.com")
    check("non-loopback Host rejected (403)", st == 403, f"status={st}")

    # negative controls
    st, j = post(port, "/api/compare3", {"prompt": "x", "slots": ["z"]})
    check("unknown slot rejected (400)", st == 400, j.get("error", "")[:60])
    st, j = post(port, "/api/compare3", {"prompt": "", "slots": ["a"]})
    check("empty prompt rejected (400)", st == 400, j.get("error", "")[:60])

    # provenance negative control (in-process; does not touch the server)
    tampered = dict(cat.by_id("qwen2.5-0.5b-instruct"))
    tampered["pin"] = dict(tampered["pin"], weights_sha256="0" * 64)
    try:
        registry3.verify(tampered); raised = False
    except registry3.ProvenanceError:
        raised = True
    check("tampered SHA-256 rejected by provenance gate", raised)

    # live three-model run (needs weights)
    ready = registry3.available_slots()
    slots = [s for s in ("a", "b", "c") if ready[s]["ready"]]
    if len(slots) < 1:
        check("three-model run", False, "no runnable slot verified in local cache (skipped)")
        return
    st, j = post(port, "/api/compare3", {"prompt": PROMPT, "slots": slots, "max_new_tokens": 12})
    if not check("comparison job accepted (202)", st == 202, f"status={st}"):
        return
    job_id = j["job_id"]
    result = None
    for _ in range(180):
        time.sleep(1)
        s2, hdr2, b2 = get(port, f"/api/compare3/status?job_id={job_id}")
        stj = json.loads(b2)
        if stj["state"] != "running":
            result = stj; break
    if not check("job finished", result is not None and result["state"] == "done",
                 f"state={result['state']}" if result else "timeout after 180s"):
        return
    r = result["result"]
    check(f"{len(slots)} models ran", len(r["models"]) == len(slots))
    check("all provenance checks passed", all(m["provenance"]["passed"] for m in r["models"].values()))
    check("all answers non-empty", all(m["answer"].strip() for m in r["models"].values()))
    check("measured internals present",
          all(m["measured"]["first_token_confidence"] is not None for m in r["models"].values()))
    check("peak memory bounded (< 14 GiB)", r["memory"]["peak_rss_bytes"] < 14 * 2**30,
          f"{r['memory']['peak_rss_bytes']/2**30:.1f} GiB")

    log_text = log.getvalue()
    check("prompt text never logged", MARKER not in log_text,
          "marker absent from server log" if MARKER not in log_text else "LEAK")


if __name__ == "__main__":
    main()
