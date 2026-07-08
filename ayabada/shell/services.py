"""Watched services: health checks for the things you self-host.

Product scope. The operator declares services in a YAML file and the
dashboard's watcher thread probes them on their intervals:

.. code-block:: yaml

    services:
      - name: website          # HTTP 2xx/3xx within timeout
        kind: http
        target: https://example.com/health
        interval: 30
      - name: postgres         # TCP connect
        kind: tcp
        target: "127.0.0.1:5432"
      - name: grafana          # docker container running
        kind: docker
        target: grafana
      - name: backup-timer     # exit code 0
        kind: command
        target: "systemctl is-active backups.timer"

Health semantics:

- ``up``        latest check succeeded within the degraded threshold
- ``degraded``  slow success, or a failure that hasn't hit the down streak
- ``down``      ``down_after`` consecutive failures (default 2)
- ``unknown``   never checked yet

Each service keeps a bounded result history (for uptime % and the latency
sparkline) and every up↔down transition is recorded in an event log.
"""

from __future__ import annotations

import shlex
import socket
import subprocess
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml

HISTORY = 60          # results kept per service (uptime window + sparkline)
EVENTS = 50           # up/down transitions kept
DEFAULT_INTERVAL = 30.0
DEFAULT_TIMEOUT = 5.0
DEFAULT_DEGRADED_MS = 1000.0
DEFAULT_DOWN_AFTER = 2

KINDS = ("http", "tcp", "docker", "command")


@dataclass(frozen=True)
class ServiceCheck:
    name: str
    kind: str
    target: str
    interval: float = DEFAULT_INTERVAL
    timeout: float = DEFAULT_TIMEOUT
    degraded_ms: float = DEFAULT_DEGRADED_MS
    down_after: int = DEFAULT_DOWN_AFTER


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    latency_ms: float | None
    detail: str


# ----------------------------------------------------------------------
# Checkers — each returns a CheckResult and never raises.
# ----------------------------------------------------------------------

def check_http(check: ServiceCheck) -> CheckResult:
    started = time.monotonic()
    try:
        request = urllib.request.Request(check.target, method="GET")
        with urllib.request.urlopen(request, timeout=check.timeout) as resp:  # noqa: S310
            latency = (time.monotonic() - started) * 1000
            return CheckResult(True, latency, f"HTTP {resp.status}")
    except Exception as exc:  # noqa: BLE001 — any failure means the check failed
        return CheckResult(False, None, f"{type(exc).__name__}: {exc}"[:120])


def check_tcp(check: ServiceCheck) -> CheckResult:
    host, _, port_text = check.target.rpartition(":")
    try:
        port = int(port_text)
    except ValueError:
        return CheckResult(False, None, f"bad target {check.target!r}; expected host:port")
    started = time.monotonic()
    try:
        with socket.create_connection((host or "127.0.0.1", port), timeout=check.timeout):
            latency = (time.monotonic() - started) * 1000
            return CheckResult(True, latency, f"connected to {check.target}")
    except OSError as exc:
        return CheckResult(False, None, f"{type(exc).__name__}: {exc}"[:120])


def check_command(check: ServiceCheck) -> CheckResult:
    started = time.monotonic()
    try:
        out = subprocess.run(
            shlex.split(check.target),
            capture_output=True,
            text=True,
            timeout=check.timeout,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return CheckResult(False, None, f"{type(exc).__name__}: {exc}"[:120])
    latency = (time.monotonic() - started) * 1000
    line = ((out.stdout or out.stderr).strip().splitlines() or [""])[0][:90]
    if out.returncode == 0:
        return CheckResult(True, latency, line or "exit 0")
    return CheckResult(False, latency, f"exit {out.returncode}" + (f": {line}" if line else ""))


def check_docker(check: ServiceCheck) -> CheckResult:
    started = time.monotonic()
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", check.target],
            capture_output=True,
            text=True,
            timeout=check.timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(False, None, f"{type(exc).__name__}: {exc}"[:120])
    latency = (time.monotonic() - started) * 1000
    if out.returncode != 0:
        detail = (out.stderr.strip().splitlines() or ["inspect failed"])[0][:90]
        return CheckResult(False, latency, detail)
    status = out.stdout.strip()
    if status == "running":
        return CheckResult(True, latency, "container running")
    return CheckResult(False, latency, f"container {status or 'unknown'}")


CHECKERS: dict[str, Callable[[ServiceCheck], CheckResult]] = {
    "http": check_http,
    "tcp": check_tcp,
    "command": check_command,
    "docker": check_docker,
}


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

class ServiceRegistry:
    """Holds checks, bounded result history, and the transition log."""

    def __init__(self, checker_map: dict[str, Callable[[ServiceCheck], CheckResult]] | None = None) -> None:
        self._checkers = checker_map or CHECKERS
        self._lock = threading.Lock()
        self._checks: dict[str, ServiceCheck] = {}
        self._history: dict[str, deque[tuple[float, bool, float | None]]] = {}
        self._last: dict[str, CheckResult] = {}
        self._last_at: dict[str, float] = {}
        self._fail_streak: dict[str, int] = {}
        self._was_down: dict[str, bool] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENTS)
        self.checks_performed = 0
        # Called with each up/down transition event, outside the lock —
        # the dashboard hooks persistence and notifications here.
        self.on_transition: Callable[[dict[str, Any]], None] | None = None

    def seed_events(self, events: list[dict[str, Any]]) -> None:
        """Restore a persisted transition log (newest-first input)."""
        with self._lock:
            for event in reversed(events):
                self._events.appendleft(event)

    def add(self, check: ServiceCheck) -> None:
        if check.kind not in self._checkers:
            raise ValueError(f"unknown service kind {check.kind!r} (expected one of {KINDS})")
        with self._lock:
            self._checks[check.name] = check
            self._history.setdefault(check.name, deque(maxlen=HISTORY))

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._checks)

    def due(self, now: float) -> list[ServiceCheck]:
        with self._lock:
            return [
                check
                for name, check in self._checks.items()
                # Never-checked services are always due — monotonic time has
                # an arbitrary epoch, so "now - 0" is meaningless.
                if name not in self._last_at or now - self._last_at[name] >= check.interval
            ]

    def run_check(self, check: ServiceCheck, now: float | None = None) -> CheckResult:
        """Execute one check (outside the lock) and record the result."""
        result = self._checkers[check.kind](check)
        now = time.monotonic() if now is None else now
        event: dict[str, Any] | None = None
        with self._lock:
            self.checks_performed += 1
            self._last[check.name] = result
            self._last_at[check.name] = now
            self._history[check.name].append((now, result.ok, result.latency_ms))
            streak = 0 if result.ok else self._fail_streak.get(check.name, 0) + 1
            self._fail_streak[check.name] = streak

            is_down = streak >= check.down_after
            was_down = self._was_down.get(check.name, False)
            if is_down != was_down:
                event = {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "service": check.name,
                    "transition": "down" if is_down else "up",
                    "detail": result.detail,
                }
                self._events.appendleft(event)
            self._was_down[check.name] = is_down
        if event is not None and self.on_transition is not None:
            try:
                self.on_transition(event)
            except Exception:  # noqa: BLE001 — a hook failure must not kill the watcher
                pass
        return result

    def snapshot(self) -> dict[str, Any]:
        """JSON view: one row per service plus the transition events."""
        with self._lock:
            now = time.monotonic()
            rows = []
            for name, check in self._checks.items():
                history = self._history[name]
                last = self._last.get(name)
                streak = self._fail_streak.get(name, 0)

                if last is None:
                    status, label = "warning", "pending"
                elif streak >= check.down_after:
                    status, label = "critical", "down"
                elif streak > 0:
                    status, label = "warning", "failing"
                elif last.latency_ms is not None and last.latency_ms > check.degraded_ms:
                    status, label = "warning", "slow"
                else:
                    status, label = "good", "up"

                uptime = (
                    100.0 * sum(1 for _, ok, _ in history if ok) / len(history)
                    if history
                    else None
                )
                checked_ago = now - self._last_at[name] if name in self._last_at else None
                rows.append(
                    {
                        "name": name,
                        "kind": check.kind,
                        "target": check.target,
                        "status": status,
                        "state_label": label,
                        "detail": last.detail if last else "first check pending",
                        "latency_ms": round(last.latency_ms, 1)
                        if last and last.latency_ms is not None
                        else None,
                        "uptime_pct": round(uptime, 1) if uptime is not None else None,
                        "checked_ago_s": round(checked_ago, 1) if checked_ago is not None else None,
                        "spark": [
                            round(lat, 1) if ok and lat is not None else None
                            for _, ok, lat in list(history)[-30:]
                        ],
                    }
                )
            return {
                "watched": rows,
                "events": list(self._events),
                "checks_performed": self.checks_performed,
            }


class ServiceWatcher(threading.Thread):
    """Runs due checks once a second; one thread for the whole registry."""

    def __init__(self, registry: ServiceRegistry, poll_seconds: float = 1.0) -> None:
        super().__init__(daemon=True, name="ayabada-service-watcher")
        self.registry = registry
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.beat_at: float | None = None

    def run(self) -> None:
        while not self.stop_event.is_set():
            for check in self.registry.due(time.monotonic()):
                self.registry.run_check(check)
            self.beat_at = time.monotonic()
            self.stop_event.wait(self.poll_seconds)


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def load_services_config(path: str) -> list[ServiceCheck]:
    """Parse the services YAML into checks; raises ValueError on bad input."""
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    entries = doc.get("services")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level 'services' list")
    checks = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry or "target" not in entry:
            raise ValueError(f"{path}: services[{i}] needs at least 'name' and 'target'")
        kind = str(entry.get("kind", "http"))
        if kind not in KINDS:
            raise ValueError(f"{path}: services[{i}] has unknown kind {kind!r}")
        checks.append(
            ServiceCheck(
                name=str(entry["name"]),
                kind=kind,
                target=str(entry["target"]),
                interval=float(entry.get("interval", DEFAULT_INTERVAL)),
                timeout=float(entry.get("timeout", DEFAULT_TIMEOUT)),
                degraded_ms=float(entry.get("degraded_ms", DEFAULT_DEGRADED_MS)),
                down_after=int(entry.get("down_after", DEFAULT_DOWN_AFTER)),
            )
        )
    return checks


def demo_checks(dashboard_port: int) -> list[ServiceCheck]:
    """Self-referential demo checks so the panel is alive without config.

    The unreachable example is deliberate: it shows what a down service
    looks like (TCP to the discard port, which nothing listens on).
    """
    return [
        ServiceCheck(
            name="dashboard-http",
            kind="http",
            target=f"http://127.0.0.1:{dashboard_port}/healthz",
            interval=5.0,
        ),
        ServiceCheck(
            name="dashboard-tcp",
            kind="tcp",
            target=f"127.0.0.1:{dashboard_port}",
            interval=5.0,
        ),
        ServiceCheck(
            name="example-down (demo)",
            kind="tcp",
            target="127.0.0.1:9",
            interval=5.0,
        ),
    ]
