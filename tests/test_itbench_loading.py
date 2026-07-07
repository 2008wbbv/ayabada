import json

from ayabada.bench.ground_truth import parse_ground_truth
from ayabada.bench.itbench import Scenario, load_snapshot

YAML = """
fault:
  - entity: {name: cart-pod, kind: Pod}
groups:
  - {id: cart-pod-1, kind: Pod, filter: ["cart-.*"], root_cause: true}
"""

EVENTS_TSV_HEADER = (
    "Timestamp\tTimestampTime\tTraceId\tSpanId\tTraceFlags\tSeverityText\t"
    "SeverityNumber\tServiceName\tBody\tResourceSchemaUrl\tResourceAttributes\t"
    "ScopeSchemaUrl\tScopeName\tScopeVersion\tScopeAttributes\tLogAttributes\tEventName\n"
)


def make_fixture(tmp_path):
    scenario_dir = tmp_path / "Test-1"
    alerts_dir = scenario_dir / "alerts"
    alerts_dir.mkdir(parents=True)

    alert = {
        "labels": {"alertname": "RequestErrorRate", "namespace": "demo", "service_name": "cart", "severity": "warning"},
        "annotations": {"description": "error rate above threshold"},
        "state": "firing",
        "activeAt": "2025-12-15T17:25:19Z",
        "value": "0.4",
    }
    # Two minute-dumps containing the same alert: loader must deduplicate.
    (alerts_dir / "alerts_at_2025-12-15T17-30-00.json").write_text(json.dumps([alert]))
    (alerts_dir / "alerts_at_2025-12-15T17-31-00.json").write_text(json.dumps([alert]))

    body = json.dumps(
        {
            "object": {
                "reason": "BackOff",
                "type": "Warning",
                "message": "restarting container",
                "count": 4,
                "involvedObject": {"kind": "Pod", "name": "cart-abc", "namespace": "demo"},
            }
        }
    )
    quoted = '"' + body.replace('"', '""') + '"'
    row = "2025-12-15 17:24:32\t2025-12-15 17:24:32\t\t\t0\t\t0\t\t" + quoted + "\t\t\t\t\t\t\t\t\n"
    (scenario_dir / "k8s_events_raw.tsv").write_text(EVENTS_TSV_HEADER + row)
    return tmp_path


def test_load_snapshot_from_fixture(tmp_path):
    cache = make_fixture(tmp_path)
    scenario = Scenario(id_aa="test-1", scenario_id="Test-1", ground_truth=parse_ground_truth(YAML, "Test-1"))
    snap = load_snapshot(scenario, cache_dir=cache)

    assert len(snap.alerts) == 1  # deduplicated
    alert = snap.alerts[0]
    assert alert.name == "RequestErrorRate"
    assert alert.labels["service_name"] == "cart"
    assert alert.value == 0.4

    assert len(snap.events) == 1
    event = snap.events[0]
    assert event["reason"] == "BackOff"
    assert event["object"] == "cart-abc"

    assert snap.raw_dir == cache / "Test-1"
    assert "cart" in snap.entity_names()
