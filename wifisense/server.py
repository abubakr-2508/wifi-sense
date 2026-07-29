"""Local dashboard server -- Python standard library only.

Deliberately dependency-free: no FastAPI, no uvicorn, no websockets. The
sensing loop runs on a background thread and publishes an immutable snapshot;
the HTTP handler serves that snapshot as JSON and the browser polls it. Fewer
moving parts means fewer things that can fail during a live demonstration.

Binds to 127.0.0.1 by default, so the dashboard is reachable from this machine
and nowhere else on the network.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .detector import MotionDetector, State
from .sources import Source

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


@dataclass
class ServiceStats:
    started_at: float = 0.0
    samples: int = 0
    dropped: int = 0
    actual_rate_hz: float = 0.0
    last_sample_at: float = 0.0


class SensingService:
    """Runs the source -> detector loop on a background thread."""

    def __init__(self, source: Source, detector: MotionDetector, poll_interval: float):
        self.source = source
        self.detector = detector
        self.poll_interval = poll_interval

        self.stats = ServiceStats()
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rate_window: list[float] = []

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self.source.open()
        self.stats.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="sensing", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.source.close()

    # -- loop -------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            tick = time.time()
            try:
                sample = self.source.read()
            except Exception:
                # A source hiccup must not kill the demo; count it and carry on.
                sample = None

            if sample is None:
                self.stats.dropped += 1
            else:
                reading = self.detector.update(
                    sample.value, sample.rssi_dbm, sample.vector
                )
                self.stats.samples += 1
                self.stats.last_sample_at = sample.timestamp
                self._track_rate(tick)
                self._publish(reading, sample)

            # Pace the loop, accounting for however long the read itself took.
            elapsed = time.time() - tick
            remaining = self.poll_interval - elapsed
            if remaining > 0:
                self._stop.wait(remaining)

    def _track_rate(self, tick: float) -> None:
        """Measured throughput -- reported alongside the nominal rate.

        The difference between nominal and actual matters: the detector's band
        calculations assume a sample rate, and if the real rate has drifted the
        frequency axis is wrong.
        """
        self._rate_window.append(tick)
        if len(self._rate_window) > 40:
            self._rate_window.pop(0)
        if len(self._rate_window) >= 2:
            span = self._rate_window[-1] - self._rate_window[0]
            if span > 0:
                self.stats.actual_rate_hz = (len(self._rate_window) - 1) / span

    def _publish(self, reading, sample) -> None:
        snap = self.detector.snapshot()
        snap["source"] = self.source.describe()
        snap["stats"] = {
            "samples": self.stats.samples,
            "dropped": self.stats.dropped,
            "uptime_s": round(time.time() - self.stats.started_at, 1),
            "nominal_rate_hz": self.source.nominal_rate_hz,
            "actual_rate_hz": round(self.stats.actual_rate_hz, 2),
        }
        snap["sample_meta"] = sample.meta
        snap["subcarriers"] = (
            [round(float(v), 2) for v in sample.vector[:64]]
            if sample.vector is not None
            else None
        )
        with self._lock:
            self._snapshot = snap

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def make_handler(service: SensingService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "wifisense/0.1"

        def log_message(self, fmt, *args):  # keep the console clean for the demo
            pass

        # -- helpers --

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json({"error": "not found"}, 404)
                return
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # -- routes --

        def do_GET(self):
            route = self.path.split("?")[0]

            if route in ("/", "/index.html"):
                self._send_file(WEB_ROOT / "index.html")
            elif route == "/api/state":
                self._send_json(service.snapshot())
            elif route == "/api/health":
                self._send_json({"ok": True, "samples": service.stats.samples})
            elif route.startswith("/static/"):
                # Resolve then confirm containment, so "..' segments cannot
                # escape the web root.
                rel = route[len("/static/") :]
                target = (WEB_ROOT / rel).resolve()
                if WEB_ROOT.resolve() in target.parents or target.parent == WEB_ROOT.resolve():
                    self._send_file(target)
                else:
                    self._send_json({"error": "forbidden"}, 403)
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            route = self.path.split("?")[0]
            if route == "/api/calibrate":
                service.detector.reset_calibration()
                self._send_json({"ok": True, "state": State.CALIBRATING.value})
            else:
                self._send_json({"error": "not found"}, 404)

    return Handler


def serve(service: SensingService, host: str = "127.0.0.1", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    return httpd
