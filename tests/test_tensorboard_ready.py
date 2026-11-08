import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import pytest

import workbench.app as application
from workbench.processes import identity


def test_tensorboard_waits_for_http_readiness_before_returning(tmp_path):
    started = time.monotonic()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if time.monotonic() - started >= .15 else 503)
            self.end_headers()
            self.wfile.write(b"{}")
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    info = {**identity(os.getpid()), "url": f"http://127.0.0.1:{server.server_port}", "log": str(tmp_path / "tensorboard.log")}
    try:
        assert application.wait_tensorboard(info, timeout=2) == info
        assert time.monotonic() - started >= .15
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def test_tensorboard_early_exit_reports_startup_log(monkeypatch):
    monkeypatch.setattr(application, "alive", lambda info: False)
    with pytest.raises(ValueError, match="tensorboard.log"):
        application.wait_tensorboard({"url": "http://127.0.0.1:1", "log": "tensorboard.log"}, timeout=1)


def test_tensorboard_timeout_is_bounded_and_does_not_return_unready_url(monkeypatch):
    monkeypatch.setattr(application, "alive", lambda info: True)
    def unavailable(*args, **kwargs):
        raise URLError("still initializing")
    monkeypatch.setattr(application, "urlopen", unavailable)
    started = time.monotonic()
    with pytest.raises(ValueError, match="tensorboard.log"):
        application.wait_tensorboard({"url": "http://127.0.0.1:1", "log": "tensorboard.log"}, timeout=.03)
    assert time.monotonic() - started < 1
