"""Doc-handoff: the markdown report a human receives after each incident.

Product scope only. The handoff is written *from* the artifacts a run already
produced (snapshot + run result) — it never feeds anything back into the
brain, so it cannot affect scored behavior.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ayabada.brain.loop import RunResult
from ayabada.snapshot import IncidentSnapshot


def render_handoff(snapshot: IncidentSnapshot, run: RunResult) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# Incident handoff — {snapshot.id}")
    lines.append("")
    lines.append(f"*Generated {now} · source: {snapshot.source}*")
    lines.append("")

    # --- outcome first ---
    if run.escalated:
        lines.append("## ⚠️ Escalated to human")
        lines.append("")
        lines.append(f"The agent halted instead of guessing: {run.escalation_reason}")
        if run.summary:
            lines.append("")
            lines.append(run.summary)
    else:
        lines.append("## Diagnosis")
        lines.append("")
        if run.predictions:
            for pred in run.predictions:
                kind = f" ({pred['kind']})" if pred.get("kind") else ""
                lines.append(f"- **{pred['name']}**{kind}")
        else:
            lines.append("- _No root-cause entity identified._")
        if run.summary:
            lines.append("")
            lines.append(run.summary)
    lines.append("")

    # --- what woke us / what the task was ---
    lines.append("## Trigger")
    lines.append("")
    if snapshot.window_start:
        lines.append(f"Window: {snapshot.window_start} → {snapshot.window_end}")
        lines.append("")
    lines.append(snapshot.description or "_no trigger description_")
    lines.append("")
    if snapshot.alerts:
        lines.append("### Alerts")
        lines.append("")
        lines.append("| Alert | Severity | Detail |")
        lines.append("|---|---|---|")
        for alert in snapshot.alerts[:20]:
            detail = alert.description.replace("|", "\\|")[:140]
            lines.append(f"| {alert.name} | {alert.severity} | {detail} |")
        if len(snapshot.alerts) > 20:
            lines.append(f"| … | | {len(snapshot.alerts) - 20} more |")
        lines.append("")

    # --- investigation record ---
    lines.append("## Investigation")
    lines.append("")
    lines.append(
        f"- Model turns: **{run.num_turns}** · tool calls: **{run.num_tool_calls}** · "
        f"stop reason: `{run.stop_reason}`"
    )
    if run.confidence_trace:
        lines.append("")
        lines.append("### Confidence trace")
        lines.append("")
        lines.append("| Turn | Confidence | Hypothesis | Rationale |")
        lines.append("|---|---|---|---|")
        for point in run.confidence_trace:
            hyp = ", ".join(h.get("name", "?") for h in point.get("hypothesis", [])) or "—"
            rationale = str(point.get("rationale", "")).replace("|", "\\|")[:100]
            lines.append(
                f"| {point.get('turn', '?')} | {point.get('confidence', 0):.2f} | {hyp} | {rationale} |"
            )
    lines.append("")
    return "\n".join(lines)
