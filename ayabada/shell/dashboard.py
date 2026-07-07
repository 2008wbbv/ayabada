"""The dashboard: a zero-dependency web UI for the self-hosted agent.

Product scope only. One stdlib HTTP server exposes:

- ``/``                  the single-page dashboard (embedded HTML/SVG/JS)
- ``/api/state``         gate state, per-metric windows (value + z), incidents
- ``/api/incident?id=…`` one incident's run detail + markdown handoff

A feed thread drives the same :class:`~ayabada.heartbeat.monitor.Heartbeat`
the CLI uses — either a synthetic demo feed (seasonal traffic with periodic
injected incidents, time-compressed) or a CSV replay. On each wake the brain
runs and the handoff document is stored with the incident, so the dashboard
shows the full loop: watch → wake → diagnose/escalate → hand off.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ayabada.brain.agent import run_confidence_loop
from ayabada.brain.loop import RunResult
from ayabada.brain.tools import SnapshotToolbox
from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
from ayabada.shell.handoff import render_handoff
from ayabada.snapshot import IncidentSnapshot

WINDOW_INTERVALS = 192  # 48h of 15-minute intervals kept for the charts


class DashboardState:
    """Thread-safe view the HTTP handlers read and the feed thread writes."""

    def __init__(self, heartbeat: Heartbeat, window: int = WINDOW_INTERVALS) -> None:
        self.heartbeat = heartbeat
        self._lock = threading.Lock()
        self._points: dict[str, deque[tuple[str, float, float | None]]] = {}
        self._window = window
        self._incidents: list[dict[str, Any]] = []
        self._incident_detail: dict[str, dict[str, Any]] = {}
        self._intervals = 0
        self._last_ts: str = ""

    # ------------------------------------------------------------------
    # Feed side
    # ------------------------------------------------------------------
    def record_interval(self, ts: datetime, observations: dict[str, float]) -> IncidentSnapshot | None:
        """Ingest one interval; returns a snapshot on the wake interval."""
        with self._lock:
            # z-scores are computed against the pre-ingest baseline — the same
            # view the detector judges — then the observation is recorded.
            z_map: dict[str, float | None] = {}
            for metric, value in observations.items():
                scored = self.heartbeat.baseline.zscore(metric, ts, value)
                z_map[metric] = round(scored[0], 3) if scored else None
            snapshot = self.heartbeat.ingest(ts, observations)
            iso = ts.isoformat()
            for metric, value in observations.items():
                self._points.setdefault(metric, deque(maxlen=self._window)).append(
                    (iso, round(value, 4), z_map[metric])
                )
            self._intervals += 1
            self._last_ts = iso
            return snapshot

    def record_incident(self, snapshot: IncidentSnapshot, run: RunResult) -> None:
        handoff = render_handoff(snapshot, run)
        with self._lock:
            self._incidents.append(
                {
                    "id": snapshot.id,
                    "at": snapshot.window_end,
                    "outcome": "escalated" if run.escalated else "diagnosed",
                    "entities": [p["name"] for p in run.predictions],
                    "turns": run.num_turns,
                    "stop_reason": run.stop_reason,
                    "final_confidence": (
                        run.confidence_trace[-1]["confidence"] if run.confidence_trace else None
                    ),
                    "metrics": sorted(
                        {a.labels.get("metric", a.name) for a in snapshot.alerts}
                    ),
                }
            )
            self._incident_detail[snapshot.id] = {
                "incident": self._incidents[-1],
                "handoff": handoff,
                "run": run.to_dict(),
            }

    # ------------------------------------------------------------------
    # HTTP side
    # ------------------------------------------------------------------
    def state_json(self) -> dict[str, Any]:
        with self._lock:
            gate = self.heartbeat.gate
            assessment = self.heartbeat.last_assessment
            decision = self.heartbeat.last_decision
            detector = self.heartbeat.detector
            return {
                "now": self._last_ts,
                "intervals": self._intervals,
                "gate": {
                    "state": gate.state.value,
                    "streak": gate.streak,
                    "wakes": len(self._incidents),
                    "reason": decision.reason if decision else "",
                    "max_z": round(assessment.max_z, 2) if assessment else 0.0,
                },
                "thresholds": {
                    "enter": detector.z_enter,
                    "severe": detector.z_severe,
                    "exit": detector.z_exit,
                    "corroboration_min": detector.corroboration_min,
                    "persistence": gate.persistence,
                },
                "metrics": {
                    metric: [list(p) for p in points]
                    for metric, points in sorted(self._points.items())
                },
                "incidents": list(reversed(self._incidents)),
            }

    def incident_json(self, incident_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._incident_detail.get(incident_id)


# ----------------------------------------------------------------------
# Feeds
# ----------------------------------------------------------------------

def seasonal_observations(ts: datetime, rng: random.Random) -> dict[str, float]:
    frac = (ts.hour * 60 + ts.minute) / 1440
    return {
        "traffic": max(1000 + 600 * math.sin(2 * math.pi * (frac - 0.3)) + rng.gauss(0, 25), 50),
        "error_rate": max(rng.gauss(0.01, 0.003), 0),
        "p95_latency_ms": max(rng.gauss(180, 12), 1),
    }


def demo_brain_factory():
    from ayabada.llm.mock import ScriptedLLM

    return ScriptedLLM(
        script=[
            {"text": "inspecting alerts", "tool": "get_alerts", "input": {}},
            '{"confidence": 0.55, "hypothesis": [{"name": "checkout-service", "kind": "Service"}], '
            '"rationale": "error rate and latency both spiked; checkout is the common dependency"}',
            {"text": "confirming", "tool": "list_entities", "input": {}},
            '{"confidence": 0.85, "hypothesis": [{"name": "checkout-service", "kind": "Service"}], '
            '"rationale": "deviations corroborate a checkout-service regression"}',
        ]
    )


class DemoFeed(threading.Thread):
    """Time-compressed synthetic feed: seasonal metrics, periodic incidents.

    One wall-clock tick = one 15-minute interval. Every ``incident_every``
    intervals an incident starts (error rate ×40, latency ×4) and runs for
    ``incident_length`` intervals, so the whole watch→wake→diagnose loop is
    visible within a couple of minutes of opening the page.
    """

    def __init__(
        self,
        state: DashboardState,
        brain_factory: Callable[[], Any] = demo_brain_factory,
        tick_seconds: float = 1.0,
        history_weeks: int = 3,
        incident_every: int = 75,
        incident_length: int = 10,
        seed: int = 11,
    ) -> None:
        super().__init__(daemon=True, name="ayabada-demo-feed")
        self.state = state
        self.brain_factory = brain_factory
        self.tick_seconds = tick_seconds
        self.history_weeks = history_weeks
        self.incident_every = incident_every
        self.incident_length = incident_length
        self.rng = random.Random(seed)
        self.ts = datetime(2026, 5, 4, 0, 0)
        self.stop_event = threading.Event()

    def prefill(self) -> None:
        """Feed seasonal history so the baseline is warm before serving."""
        for _ in range(self.history_weeks * 7 * 96):
            self.state.record_interval(self.ts, seasonal_observations(self.ts, self.rng))
            self.ts += timedelta(minutes=15)

    def run(self) -> None:
        interval = 0
        while not self.stop_event.is_set():
            observations = seasonal_observations(self.ts, self.rng)
            phase = interval % self.incident_every
            if self.incident_every - phase <= self.incident_length:
                observations["error_rate"] = 0.4 + self.rng.gauss(0, 0.02)
                observations["p95_latency_ms"] = 800 + self.rng.gauss(0, 40)
            snapshot = self.state.record_interval(self.ts, observations)
            if snapshot is not None:
                run = run_confidence_loop(self.brain_factory(), SnapshotToolbox(snapshot))
                self.state.record_incident(snapshot, run)
            self.ts += timedelta(minutes=15)
            interval += 1
            self.stop_event.wait(self.tick_seconds)


class ReplayFeed(threading.Thread):
    """Replay a wide metrics CSV (timestamp column first) through the gate."""

    def __init__(
        self,
        state: DashboardState,
        csv_path: str,
        brain_factory: Callable[[], Any] = demo_brain_factory,
        tick_seconds: float = 0.02,
    ) -> None:
        super().__init__(daemon=True, name="ayabada-replay-feed")
        self.state = state
        self.csv_path = csv_path
        self.brain_factory = brain_factory
        self.tick_seconds = tick_seconds
        self.stop_event = threading.Event()

    def run(self) -> None:
        import csv

        with open(self.csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            ts_field = reader.fieldnames[0] if reader.fieldnames else "timestamp"
            for row in reader:
                if self.stop_event.is_set():
                    return
                ts = datetime.fromisoformat(row[ts_field])
                observations = {
                    key: float(value)
                    for key, value in row.items()
                    if key != ts_field and value not in ("", None)
                }
                snapshot = self.state.record_interval(ts, observations)
                if snapshot is not None:
                    run = run_confidence_loop(self.brain_factory(), SnapshotToolbox(snapshot))
                    self.state.record_incident(snapshot, run)
                self.stop_event.wait(self.tick_seconds)


# ----------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------

def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    from ayabada.shell.dashboard_page import PAGE

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(state.state_json())
            elif parsed.path == "/api/incident":
                incident_id = parse_qs(parsed.query).get("id", [""])[0]
                detail = state.incident_json(incident_id)
                if detail is None:
                    self._send_json({"error": "unknown incident"}, status=404)
                else:
                    self._send_json(detail)
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._send(404, b"not found", "text/plain")

        def _send_json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return Handler


def serve(
    state: DashboardState,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8787,
    csv_path: str | None = None,
    tick_seconds: float = 1.0,
    log: Callable[[str], None] = print,
) -> None:
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    if csv_path:
        feed: threading.Thread = ReplayFeed(state, csv_path, tick_seconds=min(tick_seconds, 0.05))
        log(f"replaying {csv_path} through the wake gate…")
    else:
        demo = DemoFeed(state, tick_seconds=tick_seconds)
        log("warming seasonal baseline (3 simulated weeks)…")
        demo.prefill()
        feed = demo
    feed.start()
    server = serve(state, host, port)
    log(f"dashboard: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
