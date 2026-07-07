import json

from ayabada.bench.ground_truth import parse_ground_truth
from ayabada.bench.itbench import Scenario
from ayabada.bench.parity import assert_parity
from ayabada.bench.runner import DEFAULT_ARMS, build_arm_specs, run_head_to_head
from ayabada.llm.mock import ScriptedLLM
from ayabada.snapshot import Alert, IncidentSnapshot

YAML = """
fault:
  - entity: {name: cart-pod, kind: Pod}
groups:
  - {id: cart-pod-1, kind: Pod, filter: ["cart-.*"], root_cause: true}
  - {id: frontend-service-1, kind: Service, filter: ["frontend\\\\b"]}
"""


def scenario():
    return Scenario(id_aa="test-1", scenario_id="Test-1", ground_truth=parse_ground_truth(YAML, "Test-1"))


def snapshot(_scenario):
    return IncidentSnapshot(
        id="test-1",
        source="itbench",
        description="find it",
        alerts=(Alert(name="Err", labels={"service_name": "frontend"}),),
    )


def factory():
    # Script shape: to the baseline, the confidence-JSON entries are text-only
    # replies (it gets nudged and keeps going until it over-names on submit).
    # To the confidence arm they are checkpoint reports: 0.4 → continue,
    # 0.9 with a clean single-entity hypothesis → stop before the over-named
    # submission ever happens. Same model, same tools, different stopping.
    return ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            '{"confidence": 0.4, "hypothesis": [{"name": "cart-abc", "kind": "Pod"}], "rationale": "early"}',
            {"tool": "get_events", "input": {}},
            '{"confidence": 0.9, "hypothesis": [{"name": "cart-abc", "kind": "Pod"}], "rationale": "confirmed"}',
            {"tool": "submit_diagnosis", "input": {"entities": [{"name": "cart-abc", "kind": "Pod"}, {"name": "frontend", "kind": "Service"}], "summary": "s"}},
        ]
    )


def test_arm_specs_share_fingerprint_by_construction():
    specs, _probe = build_arm_specs(factory, DEFAULT_ARMS)
    assert assert_parity(*specs)
    assert specs[0].architecture != specs[1].architecture


def test_head_to_head_writes_records_and_summary(tmp_path):
    result = run_head_to_head(
        [scenario()],
        factory,
        DEFAULT_ARMS,
        snapshot_loader=snapshot,
        out_dir=tmp_path,
        log=lambda *_: None,
    )

    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert {r["arm"] for r in records} == {"baseline", "confidence"}

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["environment_fingerprint"] == result.environment_fingerprint
    base = summary["arms"]["baseline"]["aggregate"]
    conf = summary["arms"]["confidence"]["aggregate"]
    # Baseline over-names (cart + frontend): precision 0.5. The confidence arm
    # stopped at the checkpoint with the clean single-entity hypothesis.
    assert base["mean_precision"] == 0.5
    assert conf["mean_precision"] == 1.0
    assert conf["mean_score"] == 1.0
    # Both arms ran under one fingerprint recorded per arm.
    assert summary["arms"]["baseline"]["environment_fingerprint"] == result.environment_fingerprint
    assert summary["arms"]["confidence"]["environment_fingerprint"] == result.environment_fingerprint
