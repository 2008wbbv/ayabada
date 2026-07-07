import pytest

from ayabada.bench.parity import ArmSpec, ParityViolation, assert_parity

TOOLS = ({"name": "get_alerts", "input_schema": {"type": "object"}},)


def spec(**overrides):
    base = dict(
        name="arm",
        model="claude-opus-4-8",
        tool_schemas=TOOLS,
        task_instructions="diagnose",
        model_params={"max_tokens": 16000},
        architecture="baseline",
    )
    base.update(overrides)
    return ArmSpec(**base)


def test_identical_environments_pass():
    fp = assert_parity(spec(name="a"), spec(name="b", architecture="confidence"))
    assert len(fp) == 64


def test_architecture_may_differ_without_changing_fingerprint():
    a = spec(name="a", architecture="baseline")
    b = spec(name="b", architecture="confidence", architecture_params={"stop_threshold": 0.8})
    assert a.environment_fingerprint() == b.environment_fingerprint()


def test_model_difference_is_violation():
    with pytest.raises(ParityViolation):
        assert_parity(spec(name="a"), spec(name="b", model="claude-sonnet-5"))


def test_extra_tool_is_violation():
    richer = TOOLS + ({"name": "web_search", "input_schema": {"type": "object"}},)
    with pytest.raises(ParityViolation):
        assert_parity(spec(name="a"), spec(name="b", tool_schemas=richer))


def test_model_params_difference_is_violation():
    with pytest.raises(ParityViolation):
        assert_parity(spec(name="a"), spec(name="b", model_params={"max_tokens": 32000}))


def test_task_instruction_difference_is_violation():
    with pytest.raises(ParityViolation):
        assert_parity(spec(name="a"), spec(name="b", task_instructions="diagnose, with hints"))


def test_tool_order_does_not_matter():
    two = (
        {"name": "get_alerts", "input_schema": {"type": "object"}},
        {"name": "get_events", "input_schema": {"type": "object"}},
    )
    a = spec(name="a", tool_schemas=two)
    b = spec(name="b", tool_schemas=tuple(reversed(two)))
    assert a.environment_fingerprint() == b.environment_fingerprint()
