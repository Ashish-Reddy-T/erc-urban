"""scripts/sdk_healthcheck.py against a local fake SDK server (no rover)."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sdk_healthcheck  # noqa: E402


def _serve(status: int, body: bytes, delay: float = 0.0):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.paths.append(self.path)
            if delay:
                time.sleep(delay)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.paths = []
    server.daemon_threads = True
    # the timeout case writes to a socket the client already closed
    server.handle_error = lambda request, client_address: None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestSdkHealthcheck(unittest.TestCase):
    def _run(self, status, body, *extra, delay=0.0):
        server = _serve(status, body, delay)
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            code = sdk_healthcheck.main(["--url", url, *extra])
            return code, server.paths
        finally:
            server.shutdown()
            server.server_close()

    def test_ok_when_data_returns_json_object(self):
        body = json.dumps({"battery": 90, "orientation": 12}).encode()
        code, paths = self._run(200, body)
        self.assertEqual(code, 0)
        self.assertEqual(paths, ["/data"])

    def test_fails_on_http_error(self):
        code, _ = self._run(503, b'{"detail": "rtm down"}')
        self.assertNotEqual(code, 0)

    def test_fails_on_non_json_body(self):
        code, _ = self._run(200, b"<html>not json</html>")
        self.assertNotEqual(code, 0)

    def test_fails_on_timeout(self):
        t0 = time.monotonic()
        code, _ = self._run(200, b"{}", "--timeout", "0.3", delay=2.0)
        self.assertNotEqual(code, 0)
        self.assertLess(time.monotonic() - t0, 1.9)

    def test_fails_when_nothing_listens(self):
        server = _serve(200, b"{}")
        port = server.server_address[1]
        server.shutdown()
        server.server_close()
        code = sdk_healthcheck.main(
            ["--url", f"http://127.0.0.1:{port}", "--timeout", "1"])
        self.assertNotEqual(code, 0)

    def test_url_defaults_from_env(self):
        server = _serve(200, b'{"ok": 1}')
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/"
            env = {"SDK_URL": url}
            self.assertEqual(sdk_healthcheck.main([], environ=env), 0)
            self.assertEqual(server.paths, ["/data"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
