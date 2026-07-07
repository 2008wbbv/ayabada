from ayabada.brain.agent import run_confidence_loop
from ayabada.brain.loop import run_tool_loop
from ayabada.brain.policy import StoppingPolicy
from ayabada.brain.tools import SnapshotToolbox
from ayabada.llm.mock import ScriptedLLM
from ayabada.snapshot import Alert, IncidentSnapshot

CONF = '{"confidence": %s, "hypothesis": [{"name": "frontend", "kind": "Service"}], "rationale": "r"}'


def snapshot():
    return IncidentSnapshot(
        id="t",
        source="itbench",
        description="find it",
        alerts=(Alert(name="HighErr", labels={"service_name": "frontend"}),),
    )


def toolbox():
    return SnapshotToolbox(snapshot())


def test_baseline_submits():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            {"tool": "submit_diagnosis", "input": {"entities": [{"name": "frontend"}], "summary": "s"}},
        ]
    )
    r = run_tool_loop(llm, toolbox())
    assert r.stop_reason == "submitted"
    assert r.predictions == [{"name": "frontend", "kind": ""}]
    assert r.num_turns == 2 and r.num_tool_calls == 2


def test_baseline_hits_max_turns():
    llm = ScriptedLLM(script=[{"tool": "get_alerts", "input": {}}])
    r = run_tool_loop(llm, toolbox(), max_turns=5)
    assert r.stop_reason == "max_turns"
    assert r.num_turns == 5
    assert r.predictions == []


def test_baseline_nudges_on_text_only_reply():
    llm = ScriptedLLM(
        script=[
            "let me think out loud",
            {"tool": "submit_diagnosis", "input": {"entities": [{"name": "frontend"}], "summary": "s"}},
        ]
    )
    r = run_tool_loop(llm, toolbox())
    assert r.stop_reason == "submitted"
    # The nudge went into the transcript as a user message.
    nudges = [m for m in r.transcript if m["role"] == "user" and isinstance(m["content"], str) and "submit_diagnosis" in m["content"]]
    assert nudges


def test_confidence_stops_at_high_confidence():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            CONF % "0.3",
            {"tool": "get_events", "input": {}},
            CONF % "0.9",
        ]
    )
    r = run_confidence_loop(llm, toolbox())
    assert r.stop_reason.startswith("confidence_stop")
    assert r.predictions == [{"name": "frontend", "kind": "Service"}]
    assert not r.escalated
    assert len(r.confidence_trace) == 2


def test_confidence_counts_checkpoint_turns():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            CONF % "0.3",
            {"tool": "get_events", "input": {}},
            CONF % "0.9",
        ]
    )
    r = run_confidence_loop(llm, toolbox())
    # 2 investigation turns + 2 checkpoint calls, all counted.
    assert r.num_turns == 4


def test_confidence_escalates_on_stall():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            '{"confidence": 0.1, "hypothesis": [], "rationale": "no idea"}',
        ]
    )
    r = run_confidence_loop(llm, toolbox(), StoppingPolicy(patience=3))
    assert r.escalated
    assert r.stop_reason == "escalated"
    assert r.predictions == []
    assert "stalled" in r.escalation_reason


def test_confidence_honors_submit_during_checkpoint():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            {"tool": "submit_diagnosis", "input": {"entities": [{"name": "frontend"}], "summary": "sure"}},
        ]
    )
    r = run_confidence_loop(llm, toolbox())
    assert r.stop_reason == "submitted"
    assert r.predictions == [{"name": "frontend", "kind": ""}]


def test_confidence_budget_submits_best_hypothesis():
    llm = ScriptedLLM(
        script=[
            {"tool": "get_alerts", "input": {}},
            CONF % "0.5",  # decent but below stop threshold; never improves
        ]
    )
    r = run_confidence_loop(llm, toolbox(), StoppingPolicy(max_turns=6, patience=10))
    assert r.stop_reason.startswith("budget_stop")
    assert r.predictions == [{"name": "frontend", "kind": "Service"}]
