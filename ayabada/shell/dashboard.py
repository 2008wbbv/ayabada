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
import os
import platform
import random
import shutil
import subprocess
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
from ayabada.shell.notify import Notifier
from ayabada.shell.services import (
    ServiceRegistry,
    ServiceWatcher,
    demo_checks,
    load_services_config,
)
from ayabada.shell.store import RETENTION_WEEKS, StateStore
from ayabada.snapshot import IncidentSnapshot

WINDOW_INTERVALS = 192  # 48h of 15-minute intervals kept for the charts

# Action-layer tools the self-host shell can drive; probed for the services
# panel. Presence is informational — nothing in the stack requires them.
ACTION_TOOLS = ("docker", "kubectl", "git")
TOOL_PROBE_TTL = 60.0


def _probe_tool(name: str) -> tuple[bool, str]:
    """(found, detail) for one executable; never raises."""
    path = shutil.which(name)
    if path is None:
        return False, "not found on PATH"
    try:
        out = subprocess.run(
            [name, "--version"], capture_output=True, text=True, timeout=3
        )
        first = (out.stdout or out.stderr).strip().splitlines()
        return True, first[0][:90] if first else path
    except (OSError, subprocess.TimeoutExpired):
        return True, path


class ToolProbe:
    """Cached availability/version checks for the action-layer tools."""

    def __init__(
        self,
        tools: tuple[str, ...] = ACTION_TOOLS,
        ttl: float = TOOL_PROBE_TTL,
        probe: Callable[[str], tuple[bool, str]] = _probe_tool,
    ) -> None:
        self.tools = tools
        self.ttl = ttl
        self.probe = probe
        self._cache: dict[str, tuple[float, bool, str]] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        rows = []
        with self._lock:
            for name in self.tools:
                cached = self._cache.get(name)
                if cached is None or now - cached[0] > self.ttl:
                    found, detail = self.probe(name)
                    self._cache[name] = (now, found, detail)
                _, found, detail = self._cache[name]
                rows.append({"name": name, "found": found, "detail": detail})
        return rows


def _human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


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
        # --- services panel bookkeeping ---
        self.started_at = time.monotonic()
        self.address: str = ""
        self.feed_info: dict[str, Any] = {"mode": "none", "tick_seconds": 0.0}
        self.brain_info: dict[str, Any] = {"backend": "scripted demo brain", "model": None}
        self.tool_probe = ToolProbe()
        self.registry = ServiceRegistry()
        self.watcher: ServiceWatcher | None = None
        self.store: StateStore | None = None
        self.notifier: Notifier | None = None
        self.token: str | None = None
        self._replaying = False
        self._feed_beat_at: float | None = None
        self._feed_done = False
        self._brain_errors = 0
        self._last_wake_ts: str = ""

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
        if self.store is not None and not self._replaying:
            self.store.add_observations(iso, observations)
        return snapshot

    def feed_beat(self, done: bool = False) -> None:
        """Feed threads call this once per loop so liveness is observable."""
        with self._lock:
            self._feed_beat_at = time.monotonic()
            if done:
                self._feed_done = True

    def brain_error(self) -> None:
        with self._lock:
            self._brain_errors += 1

    def record_incident(self, snapshot: IncidentSnapshot, run: RunResult) -> None:
        handoff = render_handoff(snapshot, run)
        with self._lock:
            self._last_wake_ts = snapshot.window_end
            summary = {
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
            self._incidents.append(summary)
            self._incident_detail[snapshot.id] = {
                "incident": summary,
                "handoff": handoff,
                "run": run.to_dict(),
            }
        if self.store is not None:
            self.store.add_incident(
                snapshot.id, snapshot.window_end,
                {"incident": summary, "run": run.to_dict()}, handoff,
            )
        if self.notifier is not None:
            if run.escalated:
                self.notifier.send(
                    title=f"⚠ Escalated: {snapshot.id}",
                    body=run.summary or run.escalation_reason,
                    priority="high",
                    extra={"incident": summary},
                )
            else:
                entities = ", ".join(summary["entities"]) or "no entity named"
                self.notifier.send(
                    title=f"Diagnosed: {snapshot.id}",
                    body=f"Root cause: {entities} ({run.num_turns} turns). {run.summary}",
                    priority="default",
                    extra={"incident": summary},
                )

    def replay_from_store(self) -> int:
        """Rehydrate baseline, chart window, gate state and incidents.

        Observations replay through the normal ingest path (so the seasonal
        baseline and gate end up exactly as they were) but wakes during
        replay do NOT re-run the brain — those incidents are already stored.
        """
        if self.store is None:
            return 0
        self._replaying = True
        try:
            intervals = 0
            for ts_iso, observations in self.store.iter_intervals():
                self.record_interval(datetime.fromisoformat(ts_iso), observations)
                intervals += 1
            with self._lock:
                for incident_id, _at, record, handoff in self.store.load_incidents():
                    summary = record.get("incident") or {}
                    self._incidents.append(summary)
                    self._incident_detail[incident_id] = {
                        "incident": summary,
                        "handoff": handoff,
                        "run": record.get("run") or {},
                    }
                    self._last_wake_ts = summary.get("at", self._last_wake_ts)
            self.registry.seed_events(self.store.load_service_events())
            return intervals
        finally:
            self._replaying = False

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

    def services_json(self) -> dict[str, Any]:
        """Health of the self-hosted stack itself, one row per service.

        ``status`` is decided here (good/warning/critical) so the page stays
        a dumb renderer; ``state_label`` is the human word beside the dot.
        """
        tools = self.tool_probe.snapshot()  # probes outside the lock (subprocess)
        with self._lock:
            now = time.monotonic()
            rows: list[dict[str, Any]] = []

            uptime = _human_age(now - self.started_at)
            rows.append(
                {
                    "name": "Dashboard server",
                    "status": "good",
                    "state_label": "running",
                    "detail": f"{self.address} · Python {platform.python_version()}",
                    "meta": f"up {uptime}",
                }
            )

            mode = self.feed_info.get("mode", "none")
            tick = self.feed_info.get("tick_seconds", 0.0)
            if self._feed_done:
                status, label = "good", "complete"
                detail = f"{mode} feed finished · {self._intervals:,} intervals ingested"
            elif self._feed_beat_at is None:
                status, label = "warning", "starting"
                detail = f"{mode} feed has not ticked yet"
            else:
                age = now - self._feed_beat_at
                stalled = age > max(10 * tick, 15.0)
                status = "critical" if stalled else "good"
                label = "stalled" if stalled else "running"
                detail = (
                    f"{mode} feed · {self._intervals:,} intervals ingested · "
                    f"last tick {_human_age(age)} ago"
                )
            rows.append(
                {
                    "name": "Heartbeat feed",
                    "status": status,
                    "state_label": label,
                    "detail": detail,
                    "meta": (f"1 tick = 15 min ({tick:g}s real)" if mode == "demo" else ""),
                }
            )

            model = self.brain_info.get("model")
            if model:
                key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
                rows.append(
                    {
                        "name": "Agent brain",
                        "status": "good" if key_present else "critical",
                        "state_label": "ready" if key_present else "unavailable",
                        "detail": f"{model} via Anthropic API"
                        + ("" if key_present else " · ANTHROPIC_API_KEY not set"),
                        "meta": f"{self._brain_errors} run error(s)" if self._brain_errors else "",
                    }
                )
            else:
                rows.append(
                    {
                        "name": "Agent brain",
                        "status": "good",
                        "state_label": "ready",
                        "detail": "scripted demo brain (offline, no API key needed)",
                        "meta": f"{self._brain_errors} run error(s)" if self._brain_errors else "",
                    }
                )

            rows.append(
                {
                    "name": "Doc handoff",
                    "status": "good",
                    "state_label": "idle" if not self._incidents else "active",
                    "detail": f"{len(self._incidents)} handoff(s) generated"
                    + (f" · last {self._last_wake_ts[:16]}" if self._last_wake_ts else ""),
                    "meta": "",
                }
            )

            if self.store is not None:
                rows.append(
                    {
                        "name": "State store",
                        "status": "good",
                        "state_label": "persisted",
                        "detail": f"{self.store.path} · {self.store.observation_count():,} observations",
                        "meta": f"{RETENTION_WEEKS}-week retention",
                    }
                )

            if self.notifier is not None:
                stats = self.notifier.stats()
                rows.append(
                    {
                        "name": "Notifications",
                        "status": "warning" if stats["failed"] > stats["sent"] else "good",
                        "state_label": "configured",
                        "detail": f"{len(self.notifier.urls)} webhook(s), format {self.notifier.fmt}",
                        "meta": f"{stats['sent']} sent · {stats['failed']} failed",
                    }
                )

            if self.watcher is not None:
                beat = self.watcher.beat_at
                stalled = beat is None or (now - beat) > 15.0
                rows.append(
                    {
                        "name": "Service watcher",
                        "status": "warning" if stalled else "good",
                        "state_label": "stalled" if stalled else "running",
                        "detail": (
                            f"{len(self.registry.names)} service(s) watched · "
                            f"{self.registry.checks_performed} checks performed"
                        ),
                        "meta": "",
                    }
                )

            for tool in tools:
                rows.append(
                    {
                        "name": tool["name"],
                        "status": "good" if tool["found"] else "warning",
                        "state_label": "installed" if tool["found"] else "not found",
                        "detail": tool["detail"],
                        "meta": "action layer",
                    }
                )

        watched = self.registry.snapshot()  # registry has its own lock
        return {"stack": rows, **watched}


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


def run_brain(state: DashboardState, snapshot: IncidentSnapshot, brain_factory: Callable[[], Any]) -> None:
    """Run the brain on a wake and record the incident; never kill the feed.

    A brain failure (missing API key, network error) is itself an incident
    outcome — it records as an escalation with the error as the reason, so
    the wake is never silently dropped and the services panel counts it.
    """
    try:
        run = run_confidence_loop(brain_factory(), SnapshotToolbox(snapshot))
    except Exception as exc:  # noqa: BLE001 — feed must survive any brain error
        state.brain_error()
        run = RunResult(
            escalated=True,
            escalation_reason=f"brain failed to run: {type(exc).__name__}: {exc}",
            stop_reason="brain_error",
        )
        run.summary = run.escalation_reason
    state.record_incident(snapshot, run)


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
            self.state.feed_beat()
            if snapshot is not None:
                run_brain(self.state, snapshot, self.brain_factory)
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
                self.state.feed_beat()
                if snapshot is not None:
                    run_brain(self.state, snapshot, self.brain_factory)
                self.stop_event.wait(self.tick_seconds)
        self.state.feed_beat(done=True)


# ----------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------

def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    import hmac
    from http.cookies import SimpleCookie

    from ayabada.shell.dashboard_page import PAGE

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                # Liveness probe: intentionally outside auth so uptime
                # monitors (including our own self-check) can reach it.
                self._send_json({"ok": True})
                return
            if not self._authorized(parsed):
                self.send_response(401)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"unauthorized: open /?token=<token> or send Authorization: Bearer <token>")
                return
            if parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self._send_json(state.state_json())
            elif parsed.path == "/api/services":
                self._send_json(state.services_json())
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

        def _authorized(self, parsed) -> bool:
            """Bearer header, session cookie, or ?token= (which sets the cookie)."""
            token = state.token
            if not token:
                return True

            def matches(candidate: str) -> bool:
                return hmac.compare_digest(candidate.encode(), token.encode())

            auth = self.headers.get("Authorization", "")
            if auth.startswith("Bearer ") and matches(auth[len("Bearer "):].strip()):
                return True
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            if "ayabada_token" in cookie and matches(cookie["ayabada_token"].value):
                return True
            query_token = parse_qs(parsed.query).get("token", [""])[0]
            if query_token and matches(query_token):
                # Browser bootstrap: valid ?token= plants the session cookie.
                self._set_cookie = f"ayabada_token={token}; HttpOnly; SameSite=Lax; Path=/"
                return True
            return False

        _set_cookie: str | None = None

        def _send_json(self, payload: dict, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self._set_cookie:
                self.send_header("Set-Cookie", self._set_cookie)
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
    model: str | None = None,
    services_path: str | None = None,
    prometheus_path: str | None = None,
    db_path: str | None = None,
    notify_urls: list[str] | None = None,
    notify_format: str = "json",
    token: str | None = None,
    log: Callable[[str], None] = print,
) -> None:
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    state.token = token or os.environ.get("AYABADA_TOKEN") or None
    if state.token:
        log("token auth enabled")

    if notify_urls:
        state.notifier = Notifier(urls=list(notify_urls), fmt=notify_format, log=log)
        log(f"notifications: {len(notify_urls)} webhook(s), format {notify_format}")

    restored = 0
    if db_path:
        state.store = StateStore(db_path)
        restored = state.replay_from_store()
        if restored:
            log(f"restored {restored} interval(s) and "
                f"{len(state.store.load_incidents())} incident(s) from {db_path}")
        last = state.store.last_ts()
        if last:
            cutoff = datetime.fromisoformat(last) - timedelta(weeks=RETENTION_WEEKS)
            pruned = state.store.prune_before(cutoff.isoformat())
            if pruned:
                log(f"pruned {pruned} observation row(s) past {RETENTION_WEEKS}-week retention")

    if model:
        from ayabada.llm.anthropic_client import AnthropicClient

        state.brain_info = {"backend": "anthropic", "model": model}
        brain_factory: Callable[[], Any] = lambda: AnthropicClient(model=model)  # noqa: E731
    else:
        state.brain_info = {"backend": "scripted", "model": None}
        brain_factory = demo_brain_factory

    if prometheus_path:
        from ayabada.shell.prometheus import PrometheusFeed, load_prometheus_config

        config = load_prometheus_config(prometheus_path)
        state.feed_info = {"mode": "prometheus", "tick_seconds": config.interval}
        feed: threading.Thread = PrometheusFeed(state, config, brain_factory, log=log)
        log(f"polling {config.url} every {config.interval:g}s "
            f"({len(config.queries)} queries)")
    elif csv_path:
        state.feed_info = {"mode": "replay", "tick_seconds": min(tick_seconds, 0.05)}
        feed = ReplayFeed(
            state, csv_path, brain_factory=brain_factory, tick_seconds=min(tick_seconds, 0.05)
        )
        log(f"replaying {csv_path} through the wake gate…")
    else:
        state.feed_info = {"mode": "demo", "tick_seconds": tick_seconds}
        demo = DemoFeed(state, brain_factory=brain_factory, tick_seconds=tick_seconds)
        if restored:
            # Continue simulated time from where the store left off instead
            # of re-warming (the baseline came back with the replay).
            demo.ts = datetime.fromisoformat(state.store.last_ts()) + timedelta(minutes=15)
            log("baseline restored from store; demo continues without re-warming")
        else:
            log("warming seasonal baseline (3 simulated weeks)…")
            demo.prefill()
        feed = demo
    feed.start()
    server = serve(state, host, port)
    bound_port = server.server_address[1]
    state.address = f"http://{host}:{bound_port}"

    if services_path:
        checks = load_services_config(services_path)
        log(f"watching {len(checks)} service(s) from {services_path}")
    else:
        checks = demo_checks(bound_port)
        log("no --services file: watching demo self-checks")
    for check in checks:
        state.registry.add(check)

    def on_transition(event: dict) -> None:
        if state.store is not None:
            state.store.add_service_event(
                event["at"], event["service"], event["transition"], event["detail"]
            )
        if state.notifier is not None:
            arrow = "🔴 down" if event["transition"] == "down" else "🟢 up"
            state.notifier.send(
                title=f"Service {arrow}: {event['service']}",
                body=event["detail"],
                priority="high" if event["transition"] == "down" else "default",
                extra={"event": event},
            )

    state.registry.on_transition = on_transition
    state.watcher = ServiceWatcher(state.registry)
    state.watcher.start()

    log(f"dashboard: http://{host}:{bound_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
