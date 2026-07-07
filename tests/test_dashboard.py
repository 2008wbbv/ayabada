import json
import threading
import urllib.request
from datetime import datetime, timedelta

from ayabada.brain.agent import run_confidence_loop
from ayabada.brain.tools import SnapshotToolbox
from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
from ayabada.shell.dashboard import (
    DashboardState,
    DemoFeed,
    demo_brain_factory,
    seasonal_observations,
    serve,
)

START = datetime(2026, 5, 4, 0, 0)


def warmed_state(weeks: int = 3, seed: int = 3) -> tuple[DashboardState, datetime]:
    import random

    state = DashboardState(Heartbeat(HeartbeatConfig()), window=64)
    rng = random.Random(seed)
    ts = START
    for _ in range(weeks * 7 * 96):
        state.record_interval(ts, seasonal_observations(ts, rng))
        ts += timedelta(minutes=15)
    return state, ts


def drive_incident(state: DashboardState, ts: datetime, intervals: int = 10) -> datetime:
    import random

    rng = random.Random(9)
    for _ in range(intervals):
        obs = seasonal_observations(ts, rng)
        obs["error_rate"] = 0.4
        obs["p95_latency_ms"] = 850.0
        snapshot = state.record_interval(ts, obs)
        if snapshot is not None:
            run = run_confidence_loop(demo_brain_factory(), SnapshotToolbox(snapshot))
            state.record_incident(snapshot, run)
        ts += timedelta(minutes=15)
    return ts


def test_state_json_shape_and_window_bound():
    state, _ = warmed_state(weeks=1)
    data = state.state_json()
    assert data["gate"]["state"] == "asleep"
    assert data["gate"]["wakes"] == 0
    assert set(data["metrics"]) == {"traffic", "error_rate", "p95_latency_ms"}
    for points in data["metrics"].values():
        assert len(points) <= 64  # ring buffer respected
        ts, value, z = points[-1]
        assert isinstance(ts, str) and isinstance(value, float)
        assert z is None or isinstance(z, float)


def test_zscores_populate_once_history_exists():
    state, _ = warmed_state(weeks=3)
    data = state.state_json()
    last_z = [points[-1][2] for points in data["metrics"].values()]
    assert all(z is not None for z in last_z)
    assert all(abs(z) < 3 for z in last_z)  # calm data stays inside the band


def test_incident_recorded_with_handoff():
    state, ts = warmed_state()
    drive_incident(state, ts)
    data = state.state_json()
    assert data["gate"]["wakes"] == 1
    incident = data["incidents"][0]
    assert incident["outcome"] == "diagnosed"
    assert incident["entities"] == ["checkout-service"]
    detail = state.incident_json(incident["id"])
    assert detail is not None
    assert "Incident handoff" in detail["handoff"]
    assert state.incident_json("nope") is None


def test_http_endpoints():
    state, ts = warmed_state()
    drive_incident(state, ts)
    server = serve(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(f"{base}/").read().decode()
        assert "Ayabada" in html and "Seasonal deviation" in html

        data = json.loads(urllib.request.urlopen(f"{base}/api/state").read())
        assert data["gate"]["wakes"] == 1
        incident_id = data["incidents"][0]["id"]

        detail = json.loads(
            urllib.request.urlopen(f"{base}/api/incident?id={incident_id}").read()
        )
        assert detail["incident"]["id"] == incident_id
        assert "checkout-service" in detail["handoff"]

        try:
            urllib.request.urlopen(f"{base}/api/incident?id=missing")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()


def test_demo_feed_injects_and_diagnoses_incidents():
    state = DashboardState(Heartbeat(HeartbeatConfig()), window=256)
    feed = DemoFeed(state, tick_seconds=0.0, incident_every=40, incident_length=8)
    feed.prefill()
    # Run the feed loop body synchronously for a stretch that covers one
    # incident cycle, without starting the thread.
    feed.stop_event.set()  # make wait() return immediately if run() is used
    import random

    interval = 0
    for _ in range(90):
        observations = seasonal_observations(feed.ts, feed.rng)
        phase = interval % feed.incident_every
        if feed.incident_every - phase <= feed.incident_length:
            observations["error_rate"] = 0.4 + random.Random(interval).gauss(0, 0.02)
            observations["p95_latency_ms"] = 800.0
        snapshot = state.record_interval(feed.ts, observations)
        if snapshot is not None:
            run = run_confidence_loop(demo_brain_factory(), SnapshotToolbox(snapshot))
            state.record_incident(snapshot, run)
        feed.ts += timedelta(minutes=15)
        interval += 1
    data = state.state_json()
    assert data["gate"]["wakes"] >= 1
    assert all(i["outcome"] == "diagnosed" for i in data["incidents"])
