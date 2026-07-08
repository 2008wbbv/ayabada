import socket
import threading
import time

import pytest

from ayabada.shell.services import (
    CheckResult,
    ServiceCheck,
    ServiceRegistry,
    ServiceWatcher,
    check_command,
    check_tcp,
    demo_checks,
    load_services_config,
)


# ----------------------------------------------------------------------
# Checkers against real local targets
# ----------------------------------------------------------------------

def test_tcp_check_up_and_down():
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    try:
        up = check_tcp(ServiceCheck(name="s", kind="tcp", target=f"127.0.0.1:{port}"))
        assert up.ok and up.latency_ms is not None
    finally:
        listener.close()
    down = check_tcp(ServiceCheck(name="s", kind="tcp", target=f"127.0.0.1:{port}", timeout=0.5))
    assert not down.ok


def test_tcp_check_bad_target():
    result = check_tcp(ServiceCheck(name="s", kind="tcp", target="no-port-here"))
    assert not result.ok and "bad target" in result.detail


def test_command_check_exit_codes():
    ok = check_command(ServiceCheck(name="s", kind="command", target="true"))
    assert ok.ok
    bad = check_command(ServiceCheck(name="s", kind="command", target="false"))
    assert not bad.ok and "exit 1" in bad.detail
    missing = check_command(ServiceCheck(name="s", kind="command", target="no-such-cmd-xyz"))
    assert not missing.ok


# ----------------------------------------------------------------------
# Registry semantics (fake checker: fully deterministic)
# ----------------------------------------------------------------------

def scripted_registry(results):
    """Registry whose single 'svc' check pops scripted CheckResults."""
    queue = list(results)

    def fake_checker(check):
        return queue.pop(0)

    registry = ServiceRegistry(checker_map={"http": fake_checker})
    registry.add(ServiceCheck(name="svc", kind="http", target="http://x", down_after=2))
    return registry


def run_n(registry, n):
    check = ServiceCheck(name="svc", kind="http", target="http://x", down_after=2)
    for _ in range(n):
        registry.run_check(check)


def row(registry):
    return registry.snapshot()["watched"][0]


def test_pending_before_first_check():
    registry = scripted_registry([])
    assert row(registry)["state_label"] == "pending"
    assert row(registry)["status"] == "warning"


def test_up_and_uptime():
    registry = scripted_registry([CheckResult(True, 12.0, "HTTP 200")] * 4)
    run_n(registry, 4)
    r = row(registry)
    assert r["state_label"] == "up" and r["status"] == "good"
    assert r["uptime_pct"] == 100.0
    assert r["latency_ms"] == 12.0
    assert r["spark"] == [12.0] * 4


def test_single_failure_is_failing_not_down():
    registry = scripted_registry(
        [CheckResult(True, 10.0, "ok"), CheckResult(False, None, "boom")]
    )
    run_n(registry, 2)
    r = row(registry)
    assert r["state_label"] == "failing" and r["status"] == "warning"
    # no transition event yet: not down
    assert registry.snapshot()["events"] == []


def test_down_after_streak_and_recovery_events():
    registry = scripted_registry(
        [
            CheckResult(True, 10.0, "ok"),
            CheckResult(False, None, "refused"),
            CheckResult(False, None, "refused"),
            CheckResult(True, 11.0, "ok"),
        ]
    )
    run_n(registry, 3)
    r = row(registry)
    assert r["state_label"] == "down" and r["status"] == "critical"
    events = registry.snapshot()["events"]
    assert len(events) == 1 and events[0]["transition"] == "down"

    run_n(registry, 1)  # recovery
    assert row(registry)["state_label"] == "up"
    events = registry.snapshot()["events"]
    assert events[0]["transition"] == "up"  # newest first
    assert [e["transition"] for e in events] == ["up", "down"]


def test_slow_success_is_degraded():
    registry = scripted_registry([CheckResult(True, 5000.0, "HTTP 200")])
    run_n(registry, 1)
    r = row(registry)
    assert r["state_label"] == "slow" and r["status"] == "warning"


def test_sparkline_has_gaps_for_failures():
    registry = scripted_registry(
        [CheckResult(True, 10.0, "ok"), CheckResult(False, None, "x"), CheckResult(True, 12.0, "ok")]
    )
    run_n(registry, 3)
    assert row(registry)["spark"] == [10.0, None, 12.0]


def test_due_respects_intervals():
    registry = ServiceRegistry(checker_map={"http": lambda c: CheckResult(True, 1.0, "ok")})
    fast = ServiceCheck(name="fast", kind="http", target="x", interval=1.0)
    slow = ServiceCheck(name="slow", kind="http", target="x", interval=3600.0)
    registry.add(fast)
    registry.add(slow)
    now = time.monotonic()
    assert {c.name for c in registry.due(now)} == {"fast", "slow"}  # never checked
    registry.run_check(fast, now=now)
    registry.run_check(slow, now=now)
    assert registry.due(now + 2.0) == [fast]  # only fast is due again


def test_unknown_kind_rejected():
    registry = ServiceRegistry()
    with pytest.raises(ValueError):
        registry.add(ServiceCheck(name="s", kind="carrier-pigeon", target="x"))


def test_watcher_thread_runs_checks():
    registry = ServiceRegistry(checker_map={"http": lambda c: CheckResult(True, 1.0, "ok")})
    registry.add(ServiceCheck(name="svc", kind="http", target="x", interval=0.01))
    watcher = ServiceWatcher(registry, poll_seconds=0.01)
    watcher.start()
    try:
        deadline = time.monotonic() + 5
        while registry.checks_performed < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        watcher.stop_event.set()
        watcher.join(timeout=2)
    assert registry.checks_performed >= 2
    assert watcher.beat_at is not None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def test_load_services_config(tmp_path):
    path = tmp_path / "services.yml"
    path.write_text(
        """
services:
  - name: website
    kind: http
    target: https://example.com/health
    interval: 15
    degraded_ms: 500
  - name: db
    kind: tcp
    target: "127.0.0.1:5432"
"""
    )
    checks = load_services_config(str(path))
    assert [c.name for c in checks] == ["website", "db"]
    assert checks[0].interval == 15.0 and checks[0].degraded_ms == 500.0
    assert checks[1].kind == "tcp"


@pytest.mark.parametrize(
    "body",
    [
        "not-a-mapping",
        "services: {}",
        "services:\n  - name: x",                      # missing target
        "services:\n  - {name: x, target: y, kind: z}", # bad kind
    ],
)
def test_load_services_config_rejects_bad_input(tmp_path, body):
    path = tmp_path / "services.yml"
    path.write_text(body)
    with pytest.raises((ValueError, AttributeError)):
        load_services_config(str(path))


def test_demo_checks_include_down_example():
    checks = demo_checks(8787)
    kinds = {c.name: c.kind for c in checks}
    assert kinds["dashboard-http"] == "http"
    assert any("example-down" in name for name in kinds)


# ----------------------------------------------------------------------
# End-to-end through the dashboard endpoint
# ----------------------------------------------------------------------

def test_dashboard_serves_watched_services():
    import json
    import urllib.request

    from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig
    from ayabada.shell.dashboard import DashboardState, serve

    state = DashboardState(Heartbeat(HeartbeatConfig()))
    state.registry = ServiceRegistry(
        checker_map={"http": lambda c: CheckResult(True, 42.0, "HTTP 200")}
    )
    check = ServiceCheck(name="website", kind="http", target="http://x")
    state.registry.add(check)
    state.registry.run_check(check)
    state.watcher = ServiceWatcher(state.registry)
    state.watcher.beat_at = time.monotonic()

    server = serve(state, host="127.0.0.1", port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        data = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/services").read())
        assert data["watched"][0]["name"] == "website"
        assert data["watched"][0]["state_label"] == "up"
        assert data["watched"][0]["latency_ms"] == 42.0
        watcher_row = [r for r in data["stack"] if r["name"] == "Service watcher"][0]
        assert watcher_row["state_label"] == "running"
        assert "1 service(s) watched" in watcher_row["detail"]
    finally:
        server.shutdown()
