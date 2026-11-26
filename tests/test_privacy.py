"""Privacy / no-exfiltration guarantees.

Static checks that complement the functional checks in selfcheck3.py:
  * the server CSP blocks any cross-origin connection and binds to loopback;
  * the static HTML pages restrict connect-src to 'self' + the user's own loopback;
  * no shipped script fetches an external origin (so a prompt cannot be sent out);
  * the prompt is never placed in a logged string by the request logger.
"""
import re
from pathlib import Path

import server

UI = Path(__file__).resolve().parent.parent / "ui"
HTML = ["index.html", "catalog.html", "model.html"]
JS = ["api.js", "compare.js", "catalog.js", "model.js", "explain.js"]


def test_server_binds_loopback_only():
    assert server.HOST == "127.0.0.1"


def test_server_csp_blocks_cross_origin():
    csp = server.CSP
    assert "default-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "form-action 'none'" in csp


def test_server_body_limit_is_small():
    assert server.MAX_BODY <= 128 * 1024


def test_static_html_restricts_connect_src_to_self_and_loopback():
    meta = re.compile(r'http-equiv="Content-Security-Policy"\s+content="([^"]+)"', re.I)
    conn = re.compile(r"connect-src ([^;]+);")
    for name in HTML:
        html = (UI / name).read_text()
        policy = meta.search(html)
        assert policy, f"{name} has no CSP meta tag"
        m = conn.search(policy.group(1))
        assert m, f"{name} has no connect-src directive"
        srcs = m.group(1).split()
        # only self + loopback are allowed; no external origin
        allowed = {"'self'", "http://127.0.0.1:*", "http://localhost:*"}
        assert set(srcs) <= allowed, f"{name} allows unexpected connect-src: {srcs}"


def test_no_script_fetches_external_origin():
    # Any absolute http(s) URL inside a script would be a potential exfiltration path.
    url = re.compile(r"https?://(?!127\.0\.0\.1|localhost)[\w.-]+", re.I)
    for name in JS:
        text = (UI / name).read_text()
        # strip line comments before scanning
        code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
        hits = url.findall(code)
        assert not hits, f"{name} references external origin(s): {hits}"


def test_request_logger_excludes_query_and_body():
    import inspect
    src = inspect.getsource(server.Handler.log_request)
    # the logger must log the path only (no self.path with query, no body)
    assert "urlsplit(self.path).path" in src
    assert "rfile" not in src
