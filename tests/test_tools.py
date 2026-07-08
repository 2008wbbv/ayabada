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


def test_query_metrics_lists_and_summarizes_files(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    header = "metric_name\ttimestamp\tvalue\tpod_name\tnamespace\ttags\n"
    rows = []
    for i in range(20):
        rows.append(f"container_cpu_usage\t2025-12-15 17:{i:02d}:00\t{0.1 + (0.5 if i >= 15 else 0):.3f}\tcart-abc\tdemo\t{{}}\n")
        rows.append(f"container_memory_bytes\t2025-12-15 17:{i:02d}:00\t{1000 + i}\tcart-abc\tdemo\t{{}}\n")
    (metrics_dir / "pod_cart-abc_raw.tsv").write_text(header + "".join(rows))

    tb = SnapshotToolbox(snapshot(tmp_path))
    listing = tb.execute("query_metrics", {})
    assert "cart-abc" in listing

    summary = tb.execute("query_metrics", {"entity": "cart"})
    assert "container_cpu_usage" in summary and "container_memory_bytes" in summary
    assert "rising" in summary  # cpu jumps 0.1 -> 0.6 in the last quarter
    assert "n=20" in summary

    filtered = tb.execute("query_metrics", {"entity": "cart", "metric": "memory"})
    assert "container_memory_bytes" in filtered and "container_cpu_usage" not in filtered


def test_query_metrics_over_in_memory_series():
    from ayabada.snapshot import MetricSeries

    snap = IncidentSnapshot(
        id="t", source="heartbeat",
        metrics=(MetricSeries(metric="error_rate", entity="error_rate",
                              samples=tuple((float(i), 0.01) for i in range(8))),),
    )
    tb = SnapshotToolbox(snap)
    out = tb.execute("query_metrics", {"entity": "error_rate"})
    assert "error_rate" in out and "n=8" in out and "flat" in out


def test_query_metrics_no_match():
    tb = SnapshotToolbox(snapshot())
    assert "No metrics found" in tb.execute("query_metrics", {"entity": "nothing-here"})
    assert "No metric telemetry" in tb.execute("query_metrics", {})


def test_truncation():
    events = tuple(
        {"timestamp": f"t{i}", "type": "Warning", "reason": "R", "object": f"pod-{i}", "kind": "Pod", "message": "m" * 200, "count": 1}
        for i in range(200)
    )
    snap = IncidentSnapshot(id="t", source="itbench", events=events)
    out = SnapshotToolbox(snap).execute("get_events", {"limit": 200})
    assert len(out) < 9000
    assert "truncated" in out
