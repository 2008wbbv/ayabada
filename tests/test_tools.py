import pytest

from ayabada.brain.tools import SnapshotToolbox, SubmitDiagnosis
from ayabada.snapshot import Alert, IncidentSnapshot


def snapshot(tmp_path=None):
    return IncidentSnapshot(
        id="t",
        source="itbench",
        description="d",
        alerts=(
            Alert(name="HighErr", severity="critical", labels={"service_name": "cart"}),
        ),
        events=(
            {"timestamp": "t1", "type": "Warning", "reason": "BackOff", "object": "cart-abc", "kind": "Pod", "message": "restarting", "count": 3},
            {"timestamp": "t2", "type": "Normal", "reason": "Pulled", "object": "ad-xyz", "kind": "Pod", "message": "ok", "count": 1},
        ),
        raw_dir=tmp_path,
    )


def test_get_alerts_and_entities():
    tb = SnapshotToolbox(snapshot())
    assert "HighErr" in tb.execute("get_alerts", {})
    assert "cart" in tb.execute("list_entities", {})


def test_get_events_filters():
    tb = SnapshotToolbox(snapshot())
    warnings = tb.execute("get_events", {"event_type": "Warning"})
    assert "BackOff" in warnings and "Pulled" not in warnings
    by_entity = tb.execute("get_events", {"entity": "ad-"})
    assert "Pulled" in by_entity and "BackOff" not in by_entity


def test_submit_raises_signal():
    tb = SnapshotToolbox(snapshot())
    with pytest.raises(SubmitDiagnosis) as exc:
        tb.execute("submit_diagnosis", {"entities": [{"name": " cart ", "kind": "Service"}], "summary": "s"})
    assert exc.value.entities == [{"name": "cart", "kind": "Service"}]


def test_unknown_tool_and_bad_args_return_errors():
    tb = SnapshotToolbox(snapshot())
    assert tb.execute("hack_the_cluster", {}).startswith("Error")
    assert tb.execute("read_telemetry", {}).startswith("Error")


def test_telemetry_read_and_grep(tmp_path):
    (tmp_path / "metrics").mkdir()
    (tmp_path / "metrics" / "pod_cart.tsv").write_text("ts\tcpu\n1\t0.9\n2\t0.1\n")
    tb = SnapshotToolbox(snapshot(tmp_path))
    listing = tb.execute("list_telemetry_files", {})
    assert "metrics/pod_cart.tsv" in listing
    body = tb.execute("read_telemetry", {"file": "metrics/pod_cart.tsv", "max_lines": 2})
    assert "0: ts" in body
    hits = tb.execute("grep_telemetry", {"pattern": "0\\.9", "file": "metrics/pod_cart.tsv"})
    assert "0.9" in hits


def test_telemetry_path_escape_is_blocked(tmp_path):
    (tmp_path / "ok.txt").write_text("fine")
    tb = SnapshotToolbox(snapshot(tmp_path))
    out = tb.execute("read_telemetry", {"file": "../../etc/passwd"})
    assert out.startswith("Error")


def test_no_raw_dir_is_graceful():
    tb = SnapshotToolbox(snapshot())
    assert "No raw telemetry" in tb.execute("list_telemetry_files", {})
    assert tb.execute("read_telemetry", {"file": "x"}).startswith("Error")


def test_truncation():
    events = tuple(
        {"timestamp": f"t{i}", "type": "Warning", "reason": "R", "object": f"pod-{i}", "kind": "Pod", "message": "m" * 200, "count": 1}
        for i in range(200)
    )
    snap = IncidentSnapshot(id="t", source="itbench", events=events)
    out = SnapshotToolbox(snap).execute("get_events", {"limit": 200})
    assert len(out) < 9000
    assert "truncated" in out
