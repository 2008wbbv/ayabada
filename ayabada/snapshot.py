"""The one interface shared by the benchmark and the heartbeat.

An :class:`IncidentSnapshot` is a frozen, offline view of a system at the
time something went wrong: alerts, Kubernetes events, per-entity metrics and
whatever raw telemetry files exist on disk. ITBench-AA scenarios load into
this shape, and the production heartbeat packages the anomalous window it
detected into the same shape. The brain only ever sees a snapshot, so it
cannot tell (and must not care) whether it was woken by the benchmark
harness or by a live anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Alert:
    """A normalized alert (Prometheus-style)."""

    name: str
    severity: str = "warning"
    labels: dict[str, str] = field(default_factory=dict)
    description: str = ""
    active_at: str = ""
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity,
            "labels": self.labels,
            "description": self.description,
            "active_at": self.active_at,
            "value": self.value,
        }


@dataclass(frozen=True)
class MetricSeries:
    """A single metric series: (unix_ts, value) samples for one entity."""

    metric: str
    entity: str
    samples: tuple[tuple[float, float], ...] = ()
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IncidentSnapshot:
    """Frozen telemetry handed to the brain for diagnosis.

    ``source`` records provenance ("itbench" or "heartbeat") for logging and
    doc-handoff only — the brain's tool surface exposes identical views of
    either, so provenance can never leak an advantage into a scored run.
    """

    id: str
    source: str
    description: str = ""
    window_start: str = ""
    window_end: str = ""
    alerts: tuple[Alert, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    metrics: tuple[MetricSeries, ...] = ()
    # Directory holding raw snapshot files (large ITBench TSVs etc.). Tools
    # read these lazily instead of materializing hundreds of MB in memory.
    raw_dir: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def entity_names(self) -> list[str]:
        """Best-effort list of entity names visible in the telemetry."""
        names: set[str] = set()
        for alert in self.alerts:
            for key in ("service_name", "pod", "deployment", "name", "container"):
                if key in alert.labels:
                    names.add(alert.labels[key])
        for series in self.metrics:
            names.add(series.entity)
        for event in self.events:
            obj = event.get("involved_object") or event.get("object") or ""
            if obj:
                names.add(str(obj))
        return sorted(names)
