"""The confidence arm: the same loop, wrapped with calibrated stopping.

``run_confidence_loop`` uses the identical system prompt, opening message,
turn primitive and toolbox as the baseline (:func:`ayabada.brain.loop.run_tool_loop`).
The only additions are (a) a periodic confidence checkpoint — an extra
prompt/response exchange with the same stock model — and (b) the
:class:`~ayabada.brain.policy.StoppingPolicy` acting on those reports.

Honest accounting: checkpoint exchanges are counted in ``num_turns`` (they
are model calls), so any trajectory-length win reported for this arm already
pays for the mechanism's own overhead.
"""

from __future__ import annotations

from ayabada.brain.confidence import (
    CHECKPOINT_PROMPT,
    ConfidenceHistory,
    parse_confidence_report,
)
from ayabada.brain.loop import (
    SYSTEM_PROMPT,
    RunResult,
    execute_turn,
    initial_messages,
)
from ayabada.brain.policy import Decision, StoppingPolicy
from ayabada.brain.tools import SnapshotToolbox, SubmitDiagnosis
from ayabada.llm.base import LLMClient


def run_confidence_loop(
    llm: LLMClient,
    toolbox: SnapshotToolbox,
    policy: StoppingPolicy | None = None,
) -> RunResult:
    policy = policy or StoppingPolicy()
    result = RunResult()
    messages = initial_messages(toolbox)
    history = ConfidenceHistory()
    investigation_turns = 0

    while result.num_turns < policy.max_turns:
        outcome = execute_turn(llm, toolbox, SYSTEM_PROMPT, messages)
        result.num_turns += 1
        investigation_turns += 1
        result.num_tool_calls += outcome.tool_calls

        if outcome.submitted is not None:
            # The model stopped on its own — same terminal path as baseline.
            result.predictions = outcome.submitted.entities
            result.summary = outcome.submitted.summary
            result.stop_reason = "submitted"
            result.transcript = messages
            return result

        if investigation_turns % policy.checkpoint_every != 0:
            continue
        if result.num_turns >= policy.max_turns:
            break

        # --- confidence checkpoint (one extra model call) ---
        messages.append({"role": "user", "content": CHECKPOINT_PROMPT})
        response = llm.complete(system=SYSTEM_PROMPT, messages=messages, tools=toolbox.schemas())
        result.num_turns += 1
        messages.append({"role": "assistant", "content": response.content_blocks})
        if response.tool_calls:
            # The model acted instead of reporting. Execute the calls for real
            # (they are legitimate investigation actions, just mistimed) and
            # treat the checkpoint as "no signal" — except a submission, which
            # is decisive and ends the run exactly as in the baseline.
            results = []
            submitted = None
            for call in response.tool_calls:
                try:
                    output = toolbox.execute(call.name, call.input)
                except SubmitDiagnosis as signal:
                    submitted = signal
                    output = "Diagnosis submitted."
                results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": output}
                )
                result.num_tool_calls += 1
            messages.append({"role": "user", "content": results})
            if submitted is not None:
                result.predictions = submitted.entities
                result.summary = submitted.summary
                result.stop_reason = "submitted"
                result.transcript = messages
                return result
        report = parse_confidence_report(response.text)
        history.add(report)
        result.confidence_trace.append({"turn": result.num_turns, **report.to_dict()})

        decision, reason = policy.decide(history, turn=investigation_turns)
        if decision is Decision.STOP:
            best = history.best() or report
            result.predictions = list(best.hypothesis)
            result.summary = best.rationale
            result.stop_reason = f"confidence_stop: {reason}"
            result.transcript = messages
            return result
        if decision is Decision.ESCALATE:
            return _escalate(result, history, reason, messages)

        if not response.tool_calls:
            messages.append({"role": "user", "content": "Continue your investigation."})

    # Turn budget exhausted.
    decision, reason = policy.decide_at_budget(history)
    if decision is Decision.STOP:
        best = history.best()
        result.predictions = list(best.hypothesis) if best else []
        result.summary = best.rationale if best else ""
        result.stop_reason = f"budget_stop: {reason}"
        result.transcript = messages
        return result
    return _escalate(result, history, reason, messages)


def _escalate(
    result: RunResult,
    history: ConfidenceHistory,
    reason: str,
    messages: list,
) -> RunResult:
    best = history.best()
    result.escalated = True
    result.escalation_reason = reason
    result.predictions = []
    result.stop_reason = "escalated"
    if best is not None:
        names = ", ".join(h["name"] for h in best.hypothesis)
        result.summary = (
            f"Escalated to human: {reason}. Leading (unconfirmed) hypothesis "
            f"at confidence {best.confidence:.2f}: {names}. {best.rationale}"
        )
    else:
        result.summary = f"Escalated to human: {reason}. No viable hypothesis formed."
    result.transcript = messages
    return result
