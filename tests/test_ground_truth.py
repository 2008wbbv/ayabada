from ayabada.bench.ground_truth import parse_ground_truth
from ayabada.bench.itbench import load_manifest

FLAT_YAML = """
fault:
  - entity:
      name: load-generator-pod-1
      kind: Pod
    category: Change
alerts:
  - id: RequestErrorRate
    group_id: frontend-service-1
groups:
  - id: load-generator-pod-1
    kind: Pod
    filter: ["load-generator-.*"]
    namespace: otel-demo
    root_cause: true
  - id: frontend-service-1
    kind: Service
    filter: ["frontend\\\\b"]
    namespace: otel-demo
aliases:
  - [load-generator-pod-1, load-generator-service-1]
"""

WRAPPED_YAML = """
apiVersion: itbench.io/v1
kind: GroundTruth
metadata:
  name: scenario-x
spec:
  fault:
    - entity:
        changed:
          kind: ConfigMap
          name: flagd-config
      category: Configuration Setting
  groups:
    - id: flagd-config-1
      kind: ConfigMap
      filter: ["flagd-config\\\\b"]
      root_cause: true
    - id: ad-pod-1
      kind: Pod
      filter: ["adservice-.*"]
"""


def test_parse_flat_shape():
    gt = parse_ground_truth(FLAT_YAML, "s1")
    assert gt.fault_entities[0].name == "load-generator-pod-1"
    assert gt.root_cause_ids() == {gt.canonical_id("load-generator-pod-1")}


def test_parse_wrapped_shape_with_changed_entity():
    gt = parse_ground_truth(WRAPPED_YAML)
    assert gt.scenario_id == "scenario-x"
    assert gt.fault_entities[0].name == "flagd-config"
    assert gt.fault_entities[0].kind == "ConfigMap"
    assert gt.root_cause_ids() == {"flagd-config-1"}


def test_resolve_by_regex_and_kind():
    gt = parse_ground_truth(WRAPPED_YAML)
    assert gt.resolve("adservice-554b849958-lwpht", "Pod") == "ad-pod-1"
    # Kind mismatch does not match the pod group.
    assert gt.resolve("adservice-554b849958-lwpht", "Service") is None
    assert gt.resolve("totally-unknown-thing") is None


def test_alias_canonicalization():
    gt = parse_ground_truth(FLAT_YAML)
    a = gt.canonical_id("load-generator-pod-1")
    b = gt.canonical_id("load-generator-service-1")
    assert a == b


def test_full_manifest_parses_with_positive_sets():
    scenarios = load_manifest()
    assert len(scenarios) == 40
    for scenario in scenarios:
        assert scenario.ground_truth.root_cause_ids(), scenario.scenario_id
        assert scenario.ground_truth.groups, scenario.scenario_id
