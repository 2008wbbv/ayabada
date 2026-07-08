import json

import pytest

from ayabada.bench.calibration import (
    bin_points,
    calibration_from_records,
    collect_points,
    hypothesis_correct,
)
from ayabada.bench.ground_truth import parse_ground_truth
from ayabada.bench.runner import run_head_to_head
from ayabada.bench.sweep import sweep_arms, sweep_markdown, sweep_table
from ayabada.llm.mock import ScriptedLLM
from ayabada.snapshot import Alert, IncidentSnapshot

YAML = """
fault:
  - entity: {name: cart-pod, kind: Pod}
groups:
  - {id: cart-pod-1, kind: Pod, filter: ["cart-.*"], root_cause: true}
  - {id: frontend-service-1, kind: Service, filter: ["frontend\\\\b"]}
"""


def gt():
    return parse_ground_truth(YAML, "Test-1")


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def test_hypothesis_correct_requires_exact_match():
    assert hypothesis_correct(gt(), [{"name": "cart-abc", "kind": "Pod"}])
    # extra entity ruins precision -> not exactly right
    assert not hypothesis_correct(gt(), [{"name": "cart-abc"}, {"name": "frontend"}])
    assert not hypothesis_correct(gt(), [])


def test_collect_and_bin_points_ece():
    records = [
        {
            "scenario_id": "Test-1",
            "arm": "confidence",
            "run": {
                "confidence_trace": [
                    {"turn": 2, "confidence": 0.9, "parsed": True,
                     "hypothesis": [{"name": "cart-abc", "kind": "Pod"}]},
                    {"turn": 4, "confidence": 0.1, "parsed": True,
                     "hypothesis": [{"name": "frontend", "kind": "Service"}]},
                    {"turn": 6, "confidence": 0.5, "parsed": False, "hypothesis": []},
                ]
            },
        }
    ]
    points = collect_points(records, {"Test-1": gt()})
    assert len(points) == 2  # unparsed checkpoint excluded
    report = bin_points(points, n_bins=10, arm="confidence")
    assert report.n_points == 2
    # 0.9-conf point was correct (gap 0.1), 0.1-conf point was wrong (gap 0.1)
    assert report.ece == pytest.approx(0.1)
    md = report.to_markdown()
    assert "ECE 0.100" in md and "0.9–1.0" in md


def test_calibration_from_records_file(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "scenario_id": "Test-1",
                "arm": "confidence",
                "run": {"confidence_trace": [
                    {"turn": 2, "confidence": 0.8, "parsed": True,
                     "hypothesis": [{"name": "cart-x", "kind": "Pod"}]},
                ]},
            }
        )
        + "\n"
        + json.dumps({"scenario_id": "Test-1", "arm": "baseline", "run": {"confidence_trace": []}})
        + "\n"
    )
    reports = calibration_from_records(records, {"Test-1": gt()})
    assert list(reports) == ["confidence"]  # baseline contributes no points
    assert reports["confidence"].n_points == 1


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------

def test_sweep_arms_grid_and_names():
    arms = sweep_arms(stop_thresholds=(0.6, 0.9), patiences=(2, 4))
    names = [a.name for a in arms]
    assert names[0] == "baseline"
    assert "conf_s0.6_e0.35_p2_c1" in names and "conf_s0.9_e0.35_p4_c1" in names
    assert len(arms) == 1 + 4
    assert len(set(names)) == len(names)  # unique arm names


def test_sweep_arms_drops_nonsense_points():
    arms = sweep_arms(stop_thresholds=(0.3,), escalate_thresholds=(0.35, 0.2), include_baseline=False)
    # 0.35 >= 0.3 dropped; only the 0.2 escalate threshold survives
    assert [a.name for a in arms] == ["conf_s0.3_e0.2_p3_c1"]


def test_sweep_arms_empty_grid_raises():
    with pytest.raises(ValueError):
        sweep_arms(stop_thresholds=(0.3,), escalate_thresholds=(0.5,))


def test_export_predictions_bundle(tmp_path):
    from ayabada.bench.export import export_predictions

    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "scenario_id": "Test-1",
                "arm": "confidence",
                "run": {"predictions": [{"name": "cart-abc", "kind": "Pod"}],
                        "escalated": False, "summary": "s", "num_turns": 4},
                "score": {"score": 1.0, "precision": 1.0, "recall": 1.0},
            }
        )
        + "\n"
        + json.dumps({"scenario_id": "Test-1", "arm": "baseline", "run": {}, "score": {}})
        + "\n"
    )
    bundle = export_predictions(records, {"Test-1": gt()}, arm="confidence")
    assert bundle["n_scenarios"] == 1  # baseline arm excluded
    sub = bundle["submissions"][0]
    assert sub["predicted_entities"][0]["canonical_group"] == "cart-pod-1"
    assert sub["local_score"] == 1.0


def test_sweep_end_to_end_table_ordering(tmp_path):
    from ayabada.bench.itbench import Scenario

    scenario = Scenario(id_aa="t", scenario_id="Test-1", ground_truth=gt())

    def snapshot(_s):
        return IncidentSnapshot(
            id="t", source="itbench", description="find it",
            alerts=(Alert(name="Err", labels={"service_name": "cart"}),),
        )

    def factory():
        # Constant 0.7 confidence: only stop thresholds <= 0.7 stop early.
        return ScriptedLLM(script=[
            {"tool": "get_alerts", "input": {}},
            '{"confidence": 0.7, "hypothesis": [{"name": "cart-abc", "kind": "Pod"}], "rationale": "r"}',
        ])

    arms = sweep_arms(stop_thresholds=(0.6, 0.9), patiences=(3,), max_turns=10)
    result = run_head_to_head(
        [scenario], factory, arms, snapshot_loader=snapshot, out_dir=tmp_path, log=lambda *_: None
    )
    rows = sweep_table(result)
    assert rows[0]["arm"] == "conf_s0.6_e0.35_p3_c1"  # perfect score, fewest turns
    assert rows[0]["mean_score"] == 1.0
    by_name = {r["arm"]: r for r in rows}
    # The 0.9-threshold arm never stops early; budget exit submits best hypothesis.
    assert by_name["conf_s0.9_e0.35_p3_c1"]["mean_turns"] > by_name["conf_s0.6_e0.35_p3_c1"]["mean_turns"]
    md = sweep_markdown(rows)
    assert md.count("|") > 10 and "conf_s0.6" in md
