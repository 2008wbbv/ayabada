from ayabada.brain.confidence import (
    ConfidenceHistory,
    ConfidenceReport,
    parse_confidence_report,
)


def test_parse_plain_json():
    r = parse_confidence_report(
        '{"confidence": 0.7, "hypothesis": [{"name": "cart", "kind": "Service"}], "rationale": "x"}'
    )
    assert r.parsed and r.confidence == 0.7
    assert r.hypothesis[0]["name"] == "cart"


def test_parse_fenced_json_with_prose():
    text = 'Here is my report:\n```json\n{"confidence": 0.4, "hypothesis": [], "rationale": "early"}\n```\nStill looking.'
    r = parse_confidence_report(text)
    assert r.parsed and r.confidence == 0.4


def test_parse_string_hypothesis_entries():
    r = parse_confidence_report('{"confidence": 0.5, "hypothesis": ["flagd-config"]}')
    assert r.hypothesis == ({"name": "flagd-config", "kind": ""},)


def test_confidence_clamped():
    assert parse_confidence_report('{"confidence": 1.7}').confidence == 1.0
    assert parse_confidence_report('{"confidence": -2}').confidence == 0.0


def test_garbage_is_unparsed_zero():
    r = parse_confidence_report("I feel pretty good about frontend, maybe 80%?")
    assert not r.parsed and r.confidence == 0.0


def test_history_stalled():
    h = ConfidenceHistory()
    for c in (0.30, 0.31, 0.32, 0.30):
        h.add(ConfidenceReport(confidence=c))
    assert h.stalled(patience=3, epsilon=0.05)
    h.add(ConfidenceReport(confidence=0.6))
    assert not h.stalled(patience=3, epsilon=0.05)


def test_history_best_requires_hypothesis():
    h = ConfidenceHistory()
    h.add(ConfidenceReport(confidence=0.9))  # no hypothesis — can't be "best"
    h.add(ConfidenceReport(confidence=0.6, hypothesis=({"name": "cart", "kind": ""},)))
    assert h.best().confidence == 0.6
