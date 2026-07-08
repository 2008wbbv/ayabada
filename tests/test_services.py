import json
import threading
import urllib.request

from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
from ayabada.shell.dashboard import (
    DashboardState,
    ToolProbe,
    run_brain,
    serve,
)
from ayabada.snapshot import Alert, IncidentSnapshot


def fresh_state() -> DashboardState:
    state = DashboardState(Heartbeat(HeartbeatConfig()))
    state.address = "http://127.0.0.1:0"
    state.feed_info = {"mode": "demo", "tick_seconds": 1.0}
    return state


def rows_by_name(state: DashboardState) -> dict[str, dict]:
    return {row["name"]: row for row in state.services_json()["stack"]}


def test_feed_states_starting_running_stalled_complete():
    state = fresh_state()
    assert rows_by_name(state)["Heartbeat feed"]["state_label"] == "starting"

    state.feed_beat()
    row = rows_by_name(state)["Heartbeat feed"]
    assert row["state_label"] == "running" and row["status"] == "good"

    state._feed_beat_at -= 1000  # simulate a wedged feed thread
    row = rows_by_name(state)["Heartbeat feed"]
    assert row["state_label"] == "stalled" and row["status"] == "critical"

    state.feed_beat(done=True)
    assert rows_by_name(state)["Heartbeat feed"]["state_label"] == "complete"


def test_brain_row_scripted_and_anthropic(monkeypatch):
    state = fresh_state()
    row = rows_by_name(state)["Agent brain"]
    assert row["status"] == "good" and "scripted" in row["detail"]

    state.brain_info = {"backend": "anthropic", "model": "claude-opus-4-8"}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    row = rows_by_name(state)["Agent brain"]
    assert row["status"] == "critical" and "ANTHROPIC_API_KEY not set" in row["detail"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    row = rows_by_name(state)["Agent brain"]
    assert row["status"] == "good" and "claude-opus-4-8" in row["detail"]


def test_tool_probe_caches_and_reports(monkeypatch):
    calls = []

    def fake_probe(name):
        calls.append(name)
        return (name != "kubectl", f"{name} version 1.0")

    probe = ToolProbe(tools=("docker", "kubectl"), ttl=60.0, probe=fake_probe)
    first = probe.snapshot()
    second = probe.snapshot()  # inside TTL: no re-probe
    assert calls == ["docker", "kubectl"]
    assert first == second
    docker = [r for r in first if r["name"] == "docker"][0]
    kubectl = [r for r in first if r["name"] == "kubectl"][0]
    assert docker["found"] and not kubectl["found"]


def test_missing_tool_is_warning_not_critical():
    state = fresh_state()
    state.tool_probe = ToolProbe(tools=("nonexistent-tool-xyz",), probe=lambda n: (False, "not found on PATH"))
    row = rows_by_name(state)["nonexistent-tool-xyz"]
    assert row["status"] == "warning" and row["state_label"] == "not found"


def test_run_brain_survives_brain_crash_and_records_escalation():
    state = fresh_state()
    snapshot = IncidentSnapshot(
        id="hb-test-1",
        source="heartbeat",
        description="d",
        window_end="2026-05-25T16:45:00",
        alerts=(Alert(name="HeartbeatAnomaly:error_rate", labels={"metric": "error_rate"}),),
    )

    def broken_factory():
        raise RuntimeError("no API key")

    run_brain(state, snapshot, broken_factory)
    data = state.state_json()
    assert data["incidents"][0]["outcome"] == "escalated"
    assert "brain failed to run" in state.incident_json("hb-test-1")["run"]["escalation_reason"]
    assert "1 run error(s)" in rows_by_name(state)["Agent brain"]["meta"]


def test_services_endpoint():
    state = fresh_state()
    state.feed_beat()
    server = serve(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        data = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/services").read()
        )
        names = [row["name"] for row in data["stack"]]
        assert names[:4] == ["Dashboard server", "Heartbeat feed", "Agent brain", "Doc handoff"]
        assert {"docker", "kubectl", "git"} <= set(names)
        assert all(row["status"] in ("good", "warning", "critical") for row in data["stack"])
        assert data["watched"] == [] and data["events"] == []
    finally:
        server.shutdown()
