from ayabada.brain.confidence import ConfidenceHistory, ConfidenceReport
from ayabada.brain.policy import Decision, StoppingPolicy

HYP = ({"name": "cart", "kind": "Service"},)


def history(*confidences, hypothesis=HYP):
    h = ConfidenceHistory()
    for c in confidences:
        h.add(ConfidenceReport(confidence=c, hypothesis=hypothesis))
    return h


def test_stop_at_threshold():
    p = StoppingPolicy(stop_threshold=0.8, min_turns=2)
    d, _ = p.decide(history(0.85), turn=3)
    assert d is Decision.STOP


def test_no_early_stop_before_min_turns():
    p = StoppingPolicy(stop_threshold=0.8, min_turns=2)
    d, _ = p.decide(history(0.95), turn=1)
    assert d is Decision.CONTINUE


def test_no_stop_without_hypothesis():
    p = StoppingPolicy(stop_threshold=0.8, min_turns=0)
    d, _ = p.decide(history(0.9, hypothesis=()), turn=5)
    assert d is Decision.CONTINUE


def test_escalate_on_stall_below_threshold():
    p = StoppingPolicy(patience=3, escalate_threshold=0.35)
    d, reason = p.decide(history(0.1, 0.12, 0.11, 0.1), turn=6)
    assert d is Decision.ESCALATE
    assert "stalled" in reason


def test_stall_above_escalate_threshold_continues():
    p = StoppingPolicy(patience=3, escalate_threshold=0.35, stop_threshold=0.9)
    d, _ = p.decide(history(0.6, 0.61, 0.6, 0.62), turn=6)
    assert d is Decision.CONTINUE


def test_budget_submits_decent_hypothesis():
    p = StoppingPolicy(escalate_threshold=0.35)
    d, _ = p.decide_at_budget(history(0.2, 0.5))
    assert d is Decision.STOP


def test_budget_escalates_weak_hypothesis():
    p = StoppingPolicy(escalate_threshold=0.35)
    d, _ = p.decide_at_budget(history(0.1, 0.2))
    assert d is Decision.ESCALATE


def test_budget_escalates_with_no_reports():
    p = StoppingPolicy()
    d, _ = p.decide_at_budget(ConfidenceHistory())
    assert d is Decision.ESCALATE
