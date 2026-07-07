"""ITBench-AA substrate: manifest, scenario download, snapshot loading.

The manifest (``sre/data.jsonl``, 40 public scenarios) is small and vendored
under ``data/itbench/``. Full telemetry snapshots are large (metrics/logs/
traces are 20–280 MB LFS files per scenario), so they are fetched on demand
into a local cache directory and loaded lazily.

Dataset: https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA
(CC-BY-4.0, Artificial Analysis release of IBM ITBench public SRE scenarios).
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ayabada.bench.ground_truth import GroundTruth, parse_ground_truth
from ayabada.snapshot import Alert, IncidentSnapshot

HF_BASE = "https://huggingface.co/datasets/ArtificialAnalysis/ITBench-AA/resolve/main"
HF_API_TREE = "https://huggingface.co/api/datasets/ArtificialAnalysis/ITBench-AA/tree/main"

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "itbench" / "data.jsonl"
DEFAULT_CACHE = Path.home() / ".cache" / "ayabada" / "itbench"

# The task prompt every agent (both arms) receives. Fixed text: part of the
# shared harness, never varied between arms.
TASK_INSTRUCTIONS = (
    "You are diagnosing a Kubernetes incident from an offline snapshot of the "
    "affected cluster (alerts, events, metrics). Identify the entity or "
    "entities (Pod, Service, Deployment, ConfigMap, ...) responsible for the "
    "failure — the root cause, not the symptoms. Name only entities you "
    "believe caused the incident: extra entities beyond the true root cause "
    "reduce your score."
)


@dataclass(frozen=True)
class Scenario:
    id_aa: str
    scenario_id: str
    ground_truth: GroundTruth

    @property
    def name(self) -> str:
        return self.scenario_id


def load_manifest(path: Path | None = None) -> list[Scenario]:
    """Read the vendored manifest into Scenario objects (ground truth included)."""
    path = path or DEFAULT_MANIFEST
    scenarios: list[Scenario] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt = parse_ground_truth(row["ground_truth_yaml"], row["scenario_id"])
            scenarios.append(
                Scenario(id_aa=row["id_aa"], scenario_id=row["scenario_id"], ground_truth=gt)
            )
    return scenarios


# ----------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read()


def _list_files(subpath: str) -> list[dict]:
    data = json.loads(_http_get(f"{HF_API_TREE}/{subpath}").decode())
    return data if isinstance(data, list) else []


def fetch_scenario(
    scenario_id: str,
    cache_dir: Path | None = None,
    include_metrics: bool = False,
    log=print,
) -> Path:
    """Download a scenario's snapshot files into the cache; return its dir.

    By default fetches the small files (per-minute alerts JSON, ground truth,
    k8s events TSV). ``include_metrics=True`` also pulls the per-pod metric
    TSVs (~20 MB each). The giant otel logs/traces TSVs are never fetched
    automatically — point tools at them manually if an experiment needs them.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    dest = cache_dir / scenario_id
    dest.mkdir(parents=True, exist_ok=True)

    wanted: list[str] = []
    for entry in _list_files(f"sre/{scenario_id}"):
        path = entry["path"]
        if path.endswith(("ground_truth.yaml", "k8s_events_raw.tsv")):
            wanted.append(path)
    for entry in _list_files(f"sre/{scenario_id}/alerts"):
        wanted.append(entry["path"])
    if include_metrics:
        for entry in _list_files(f"sre/{scenario_id}/metrics"):
            wanted.append(entry["path"])

    for path in wanted:
        rel = Path(path).relative_to(f"sre/{scenario_id}")
        target = dest / rel
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log(f"fetch {path}")
        target.write_bytes(_http_get(f"{HF_BASE}/{path}"))
    return dest


# ----------------------------------------------------------------------
# Snapshot loading
# ----------------------------------------------------------------------

def _load_alerts(alerts_dir: Path) -> list[Alert]:
    """Merge the per-minute alert dumps, deduplicating identical alerts."""
    seen: dict[tuple, Alert] = {}
    for path in sorted(alerts_dir.glob("alerts_at_*.json")):
        try:
            items = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for item in items:
            labels = dict(item.get("labels") or {})
            annotations = item.get("annotations") or {}
            key = (labels.get("alertname", ""), tuple(sorted(labels.items())))
            if key in seen:
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                value = None
            seen[key] = Alert(
                name=labels.get("alertname", "unknown"),
                severity=labels.get("severity", "warning"),
                labels={k: v for k, v in labels.items() if k != "alertname"},
                description=str(annotations.get("description", annotations.get("summary", ""))),
                active_at=str(item.get("activeAt", "")),
                value=value,
            )
    return list(seen.values())


def _load_events(path: Path, limit: int = 5000) -> list[dict]:
    """Parse the k8s events TSV (OTLP log rows whose Body is JSON)."""
    events: list[dict] = []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if len(events) >= limit:
                break
            body = row.get("Body") or ""
            try:
                obj = json.loads(body).get("object", {})
            except (json.JSONDecodeError, AttributeError):
                continue
            involved = obj.get("involvedObject") or {}
            if not obj.get("message") and not involved:
                continue
            events.append(
                {
                    "timestamp": row.get("Timestamp", ""),
                    "reason": obj.get("reason", ""),
                    "type": obj.get("type", ""),
                    "message": obj.get("message", ""),
                    "kind": involved.get("kind", ""),
                    "object": involved.get("name", ""),
                    "namespace": involved.get("namespace", ""),
                    "count": obj.get("count", 1),
                }
            )
    return events


def load_snapshot(scenario: Scenario, cache_dir: Path | None = None) -> IncidentSnapshot:
    """Build the IncidentSnapshot for a fetched scenario."""
    cache_dir = cache_dir or DEFAULT_CACHE
    raw = cache_dir / scenario.scenario_id
    if not raw.exists():
        raise FileNotFoundError(
            f"Scenario {scenario.scenario_id} not fetched; run fetch_scenario() first"
        )

    alerts = _load_alerts(raw / "alerts") if (raw / "alerts").exists() else []
    events_path = raw / "k8s_events_raw.tsv"
    events = _load_events(events_path) if events_path.exists() else []

    return IncidentSnapshot(
        id=scenario.id_aa,
        source="itbench",
        description=TASK_INSTRUCTIONS,
        alerts=tuple(alerts),
        events=tuple(events),
        raw_dir=raw,
    )
