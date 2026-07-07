from ayabada.bench.ground_truth import parse_ground_truth
from ayabada.bench.scoring import Prediction, aggregate, score_task

YAML = """
fault:
  - entity: {name: flagd-config, kind: ConfigMap}
groups:
  - {id: flagd-config-1, kind: ConfigMap, filter: ["flagd-config\\\\b"], root_cause: true}
  - {id: ad-pod-1, kind: Pod, filter: ["adservice-.*"]}
  - {id: frontend-service-1, kind: Service, filter: ["frontend\\\\b"]}
"""


def gt():
    return parse_ground_truth(YAML, "s")


def test_exact_hit_scores_one():
    s = score_task(gt(), [Prediction("flagd-config", "ConfigMap")])
    assert (s.precision, s.recall, s.score) == (1.0, 1.0, 1.0)


def test_extra_entity_halves_precision_and_score():
    s = score_task(gt(), [Prediction("flagd-config", "ConfigMap"), Prediction("frontend", "Service")])
    assert s.precision == 0.5
    assert s.recall == 1.0
    assert s.score == 0.5  # recall gate met, precision counts


def test_missed_root_cause_gates_score_to_zero():
    s = score_task(gt(), [Prediction("adservice-abc", "Pod")])
    assert s.recall == 0.0
    assert s.score == 0.0
    assert s.false_negatives == ["flagd-config-1"]


def test_unknown_entity_is_false_positive():
    s = score_task(gt(), [Prediction("flagd-config"), Prediction("does-not-exist")])
    assert "does-not-exist" in s.false_positives
    assert s.precision == 0.5


def test_duplicate_aliases_count_once():
    s = score_task(gt(), [Prediction("flagd-config"), Prediction("flagd-config", "ConfigMap")])
    assert s.precision == 1.0 and s.score == 1.0


def test_escalation_scores_zero_but_is_tracked():
    s = score_task(gt(), [], escalated=True, stop_reason="escalated")
    assert s.score == 0.0 and s.escalated


def test_relaxed_recall_gate():
    two_cause_yaml = YAML.replace(
        'root_cause: true}',
        'root_cause: true}\n  - {id: extra-1, kind: Pod, filter: ["extra-.*"], root_cause: true}',
        1,
    )
    gt2 = parse_ground_truth(two_cause_yaml, "s2")
    assert len(gt2.root_cause_ids()) == 2
    s = score_task(gt2, [Prediction("flagd-config")], recall_gate=0.5)
    assert s.recall == 0.5
    assert s.score == 1.0  # gate met at 0.5, precision is perfect


def test_aggregate():
    scores = [
        score_task(gt(), [Prediction("flagd-config")], num_turns=4),
        score_task(gt(), [], escalated=True, num_turns=8),
    ]
    agg = aggregate(scores)
    assert agg["n_tasks"] == 2
    assert agg["mean_score"] == 0.5
    assert agg["escalation_rate"] == 0.5
    assert agg["mean_turns"] == 6
