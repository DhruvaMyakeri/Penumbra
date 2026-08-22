"""Live dashboard for a garage run.

Serves the run directory and streams `state.json` to the browser over
Server-Sent Events, so generations, gate verdicts and statistical results appear as
they happen rather than at the end. The runner writes state after every step; this
watches the file's mtime and pushes.

Deliberately a stdlib `ThreadingHTTPServer`: the run is a local process, the audience
is one engineer watching their own experiment, and a framework here would be weight
without benefit.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("penumbra.ui")


class _Handler(SimpleHTTPRequestHandler):
    """Static files from the run directory, plus /events for live state."""

    def __init__(self, *args, run_dir: Path, **kw):
        self.run_dir = run_dir
        super().__init__(*args, directory=str(run_dir), **kw)

    def log_message(self, fmt, *args):  # noqa: A003 - quiet by default
        log.debug(fmt, *args)

    def handle_one_request(self):
        """Swallow client-disconnect noise.

        A long-lived SSE stream is disconnected constantly - every refresh, tab close
        and navigation - and the default handler prints a full traceback for each one.
        They are not errors in the run, and burying real failures under them is worse
        than losing them.
        """
        try:
            super().handle_one_request()
        except OSError:
            self.close_connection = True

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/events"):
            return self._events()
        if self.path in ("/", "/index.html"):
            return self._page()
        return super().do_GET()

    def _page(self):
        html = (Path(__file__).parent / "dashboard.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html)

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        state_path = self.run_dir / "state.json"
        last_sent = None
        try:
            while True:
                try:
                    raw = state_path.read_text(encoding="utf-8")
                except (OSError, ValueError):
                    raw = None
                if raw and raw != last_sent:
                    # A partially written file would break the client's JSON.parse;
                    # skip until it parses rather than pushing a truncated frame.
                    try:
                        json.loads(raw)
                    except ValueError:
                        time.sleep(0.2)
                        continue
                    last_sent = raw
                    payload = "".join(f"data: {line}\n" for line in raw.splitlines())
                    self.wfile.write((payload + "\n").encode())
                    self.wfile.flush()
                else:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                time.sleep(0.6)
        except OSError:
            # The browser navigating away, refreshing, or closing the tab aborts the
            # stream. On Windows that surfaces as ConnectionAbortedError (WinError
            # 10053) rather than BrokenPipeError, so catch the shared base: every
            # socket error here means the same thing - this client is gone.
            return


class _Server(ThreadingHTTPServer):
    # HTTPServer sets allow_reuse_address, which on Windows lets a second server bind
    # a port another process already holds. Both then answer unpredictably, and a run
    # silently serves a *previous* run's directory - media 404s, state is stale, and
    # the dashboard looks broken for reasons nothing in the log explains. Refusing to
    # share the port turns that into an immediate, readable error.
    allow_reuse_address = False


def serve(run_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Start the dashboard in a background thread and return the server.

    Raises if the port is already held, naming the likely cause, rather than binding
    alongside an existing server.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(_Handler, run_dir=run_dir)
    try:
        httpd = _Server(("127.0.0.1", port), handler)
    except OSError as exc:
        raise RuntimeError(
            f"port {port} is already in use - another garage run is probably still "
            f"serving an older directory. Stop it, or pass --port. Serving alongside "
            f"it would show you that run's results instead of this one."
        ) from exc
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="penumbra-ui")
    thread.start()
    log.info("dashboard on http://127.0.0.1:%d", port)
    return httpd
