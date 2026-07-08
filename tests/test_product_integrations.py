"""Prometheus feed, notifications, persistence, and token auth."""

import json
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
from ayabada.shell.dashboard import DashboardState, serve
from ayabada.shell.notify import Notifier
from ayabada.shell.prometheus import (
    PrometheusConfig,
    PrometheusPoller,
    load_prometheus_config,
    query_instant,
)
from ayabada.shell.store import StateStore

START = datetime(2026, 5, 4, 0, 0)


# ----------------------------------------------------------------------
# Shared capture server
# ----------------------------------------------------------------------

class CaptureServer:
    """Tiny HTTP server that records requests and serves canned responses."""

    def __init__(self, responder=None):
        self.requests: list[dict] = []
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self, body: bytes) -> None:
                capture.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body.decode(errors="replace"),
                    }
                )
                status, payload = (200, b"{}")
                if responder is not None:
                    status, payload = responder(self.path)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                self._handle(self.rfile.read(length))

            def do_GET(self):  # noqa: N802
                self._handle(b"")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def prom_response(value="1.5", result_type="vector", n=1):
    if result_type == "scalar":
        data = {"resultType": "scalar", "result": [1e9, value]}
    else:
        data = {
            "resultType": "vector",
            "result": [{"metric": {}, "value": [1e9, value]} for _ in range(n)],
        }
    return json.dumps({"status": "success", "data": data}).encode()


# ----------------------------------------------------------------------
# Prometheus
# ----------------------------------------------------------------------

def test_query_instant_vector_and_scalar():
    server = CaptureServer(lambda path: (200, prom_response("2.5")))
    try:
        assert query_instant(server.url, "up") == 2.5
    finally:
        server.close()
    server = CaptureServer(lambda path: (200, prom_response("7", "scalar")))
    try:
        assert query_instant(server.url, "scalar(1)") == 7.0
    finally:
        server.close()


def test_query_instant_sums_multielement_vectors():
    server = CaptureServer(lambda path: (200, prom_response("2.0", n=3)))
    try:
        assert query_instant(server.url, "rate(x[5m])") == 6.0
    finally:
        server.close()


def test_query_instant_raises_on_empty_and_error():
    server = CaptureServer(lambda path: (200, prom_response(n=0)))
    try:
        with pytest.raises(ValueError):
            query_instant(server.url, "nothing")
    finally:
        server.close()
    server = CaptureServer(
        lambda path: (200, json.dumps({"status": "error", "error": "bad expr"}).encode())
    )
    try:
        with pytest.raises(ValueError):
            query_instant(server.url, "bad(")
    finally:
        server.close()


def test_poller_partial_results_on_failures():
    def responder(path):
        if "good_metric" in urllib.request.unquote(path):
            return 200, prom_response("3.0")
        return 500, b"boom"

    server = CaptureServer(responder)
    logs = []
    try:
        config = PrometheusConfig(
            url=server.url,
            queries={"good_metric": "good_metric", "bad_metric": "bad_metric"},
        )
        poller = PrometheusPoller(config, log=logs.append)
        observations = poller.poll()
        assert observations == {"good_metric": 3.0}
        assert poller.consecutive_failures == 0
        assert any("bad_metric" in line for line in logs)
    finally:
        server.close()


def test_load_prometheus_config(tmp_path):
    path = tmp_path / "prom.yml"
    path.write_text(
        "prometheus:\n  url: http://prom:9090/\n  interval: 60\n"
        "queries:\n  traffic: sum(rate(x[5m]))\n"
    )
    config = load_prometheus_config(str(path))
    assert config.url == "http://prom:9090"  # trailing slash stripped
    assert config.interval == 60.0
    assert config.queries == {"traffic": "sum(rate(x[5m]))"}
    bad = tmp_path / "bad.yml"
    bad.write_text("queries: {}")
    with pytest.raises(ValueError):
        load_prometheus_config(str(bad))


# ----------------------------------------------------------------------
# Notifications
# ----------------------------------------------------------------------

def test_notifier_formats():
    server = CaptureServer()
    try:
        for fmt in ("json", "ntfy", "slack"):
            Notifier(urls=[server.url], fmt=fmt, log=lambda *_: None).send(
                "Title X", "body text", priority="high"
            )
        json_req, ntfy_req, slack_req = server.requests
        payload = json.loads(json_req["body"])
        assert payload["title"] == "Title X" and payload["priority"] == "high"
        assert ntfy_req["headers"]["Title"] == "Title X"
        assert ntfy_req["body"] == "body text"
        assert "*Title X*" in json.loads(slack_req["body"])["text"]
    finally:
        server.close()


def test_notifier_counts_failures_and_never_raises():
    notifier = Notifier(urls=["http://127.0.0.1:9/unreachable"], log=lambda *_: None)
    assert notifier.send("t", "b") == 0
    assert notifier.stats() == {"sent": 0, "failed": 1}


def test_notifier_rejects_unknown_format():
    with pytest.raises(ValueError):
        Notifier(urls=["http://x"], fmt="smoke-signals")


def test_wake_outcome_triggers_notification():
    from ayabada.shell.dashboard import run_brain, demo_brain_factory
    from ayabada.snapshot import Alert, IncidentSnapshot

    server = CaptureServer()
    try:
        state = DashboardState(Heartbeat(HeartbeatConfig()))
        state.notifier = Notifier(urls=[server.url], log=lambda *_: None)
        snapshot = IncidentSnapshot(
            id="hb-1", source="heartbeat", description="d", window_end="2026-05-25T16:45:00",
            alerts=(Alert(name="HeartbeatAnomaly:error_rate", labels={"metric": "error_rate"}),),
        )
        run_brain(state, snapshot, demo_brain_factory)
        assert len(server.requests) == 1
        payload = json.loads(server.requests[0]["body"])
        assert payload["title"].startswith("Diagnosed: hb-1")
        assert "checkout-service" in payload["body"]
    finally:
        server.close()


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def feed_state(state: DashboardState, weeks: int = 1) -> datetime:
    import random

    rng = random.Random(3)
    ts = START
    for _ in range(weeks * 7 * 96):
        from ayabada.shell.dashboard import seasonal_observations

        state.record_interval(ts, seasonal_observations(ts, rng))
        ts += timedelta(minutes=15)
    return ts


def test_store_roundtrip_and_restart_recovery(tmp_path):
    from ayabada.shell.dashboard import demo_brain_factory, run_brain, seasonal_observations
    import random

    db = tmp_path / "ayabada.db"

    # First life: observe a week, then an incident.
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    state.store = StateStore(db)
    ts = feed_state(state, weeks=1)
    rng = random.Random(5)
    woke = False
    for _ in range(10):
        obs = seasonal_observations(ts, rng)
        obs["error_rate"] = 0.4
        obs["p95_latency_ms"] = 850.0
        snapshot = state.record_interval(ts, obs)
        if snapshot is not None:
            run_brain(state, snapshot, demo_brain_factory)
            woke = True
        ts += timedelta(minutes=15)
    assert woke
    first_life = state.state_json()
    state.store.close()

    # Second life: fresh objects, same file.
    state2 = DashboardState(Heartbeat(HeartbeatConfig()))
    state2.store = StateStore(db)
    restored = state2.replay_from_store()
    assert restored == state2.state_json()["intervals"] > 600

    second_life = state2.state_json()
    assert second_life["incidents"] == first_life["incidents"]
    assert second_life["gate"]["wakes"] == 1
    # The baseline is warm again: z-scores are judged, not None.
    last_points = [pts[-1][2] for pts in second_life["metrics"].values()]
    assert all(z is not None for z in last_points)
    # Incident detail (handoff) survived too.
    incident_id = second_life["incidents"][0]["id"]
    assert "Incident handoff" in state2.incident_json(incident_id)["handoff"]
    # Replay itself must not have duplicated rows in the store.
    assert state2.store.observation_count() == restored * 3


def test_store_prune(tmp_path):
    store = StateStore(tmp_path / "x.db")
    store.add_observations("2026-01-01T00:00:00", {"m": 1.0})
    store.add_observations("2026-03-01T00:00:00", {"m": 2.0})
    assert store.prune_before("2026-02-01T00:00:00") == 1
    assert store.observation_count() == 1
    assert store.last_ts() == "2026-03-01T00:00:00"


def test_service_events_persist(tmp_path):
    store = StateStore(tmp_path / "x.db")
    store.add_service_event("2026-05-01 10:00:00", "web", "down", "refused")
    store.add_service_event("2026-05-01 10:05:00", "web", "up", "HTTP 200")
    events = store.load_service_events()
    assert [e["transition"] for e in events] == ["up", "down"]  # newest first


# ----------------------------------------------------------------------
# Token auth
# ----------------------------------------------------------------------

def test_token_auth_flow():
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    state.token = "s3cret"
    server = serve(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{base}/api/state")
        assert exc.value.code == 401

        # /healthz stays reachable without auth (self-check + uptime monitors).
        health = json.loads(urllib.request.urlopen(f"{base}/healthz").read())
        assert health == {"ok": True}

        request = urllib.request.Request(
            f"{base}/api/state", headers={"Authorization": "Bearer s3cret"}
        )
        assert urllib.request.urlopen(request).status == 200

        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/api/state", headers={"Authorization": "Bearer wrong"}
                )
            )

        # ?token= bootstraps a session cookie for the browser.
        response = urllib.request.urlopen(f"{base}/?token=s3cret")
        assert response.status == 200
        cookie = response.headers.get("Set-Cookie", "")
        assert "ayabada_token=s3cret" in cookie and "HttpOnly" in cookie

        cookie_request = urllib.request.Request(
            f"{base}/api/state", headers={"Cookie": "ayabada_token=s3cret"}
        )
        assert urllib.request.urlopen(cookie_request).status == 200
    finally:
        server.shutdown()


def test_no_token_means_open():
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    server = serve(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state").status == 200
    finally:
        server.shutdown()
