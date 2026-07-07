"""The standard tool-use loop — shared machinery for both arms.

``run_tool_loop`` *is* the baseline agent: model + tools + loop until it
submits a diagnosis or hits the turn cap. The confidence arm reuses the same
turn primitive (:func:`execute_turn`) and the same nudge text, so the only
difference between arms is the wrapper logic in :mod:`ayabada.brain.agent`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ayabada.brain.tools import SnapshotToolbox, SubmitDiagnosis
from ayabada.llm.base import LLMClient

SYSTEM_PROMPT = (
    "You are an SRE agent diagnosing a Kubernetes incident from an offline "
    "snapshot. Investigate with the tools provided. When you have identified "
    "the root cause, call submit_diagnosis with ONLY the responsible "
    "entities — naming symptom entities alongside the root cause reduces the "
    "score. If a tool errors, adjust your arguments and continue."
)

NUDGE = (
    "Continue investigating with the tools, or call submit_diagnosis if you "
    "have identified the root cause."
)

DEFAULT_MAX_TURNS = 30


@dataclass
class RunResult:
    """Outcome of one agent run on one snapshot."""

    predictions: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    escalated: bool = False
    escalation_reason: str = ""
    stop_reason: str = ""
    num_turns: int = 0
    num_tool_calls: int = 0
    confidence_trace: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions,
            "summary": self.summary,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "stop_reason": self.stop_reason,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "confidence_trace": self.confidence_trace,
        }


@dataclass
class TurnOutcome:
    """What one model turn produced."""

    text: str
    tool_calls: int
    submitted: SubmitDiagnosis | None = None


def execute_turn(
    llm: LLMClient,
    toolbox: SnapshotToolbox,
    system: str,
    messages: list[dict[str, Any]],
) -> TurnOutcome:
    """One model call + tool execution, appending to ``messages`` in place.

    Returns the turn outcome; when the model calls ``submit_diagnosis`` the
    signal is captured and returned rather than raised.
    """
    response = llm.complete(system=system, messages=messages, tools=toolbox.schemas())
    messages.append({"role": "assistant", "content": response.content_blocks})

    if not response.tool_calls:
        messages.append({"role": "user", "content": NUDGE})
        return TurnOutcome(text=response.text, tool_calls=0)

    submitted: SubmitDiagnosis | None = None
    results: list[dict[str, Any]] = []
    for call in response.tool_calls:
        if submitted is not None:
            # The run is over; acknowledge trailing parallel calls so the
            # transcript stays well-formed.
            results.append(_tool_result(call.id, "Run ended: diagnosis already submitted."))
            continue
        try:
            output = toolbox.execute(call.name, call.input)
        except SubmitDiagnosis as signal:
            submitted = signal
            results.append(_tool_result(call.id, "Diagnosis submitted."))
            continue
        results.append(_tool_result(call.id, output))
    messages.append({"role": "user", "content": results})
    return TurnOutcome(text=response.text, tool_calls=len(response.tool_calls), submitted=submitted)


def _tool_result(tool_use_id: str, content: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def initial_messages(toolbox: SnapshotToolbox) -> list[dict[str, Any]]:
    """The opening user turn — identical for both arms."""
    return [
        {
            "role": "user",
            "content": (
                f"{toolbox.snapshot.description}\n\n"
                f"Incident snapshot id: {toolbox.snapshot.id}. "
                "Begin your investigation."
            ),
        }
    ]


def run_tool_loop(
    llm: LLMClient,
    toolbox: SnapshotToolbox,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> RunResult:
    """The vanilla baseline: loop until submission or the turn cap."""
    result = RunResult()
    messages = initial_messages(toolbox)

    for _ in range(max_turns):
        outcome = execute_turn(llm, toolbox, SYSTEM_PROMPT, messages)
        result.num_turns += 1
        result.num_tool_calls += outcome.tool_calls
        if outcome.submitted is not None:
            result.predictions = outcome.submitted.entities
            result.summary = outcome.submitted.summary
            result.stop_reason = "submitted"
            break
    else:
        result.stop_reason = "max_turns"

    result.transcript = messages
    return result
