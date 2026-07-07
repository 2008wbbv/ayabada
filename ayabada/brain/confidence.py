"""Confidence elicitation: the checkpoint the wrapper injects between turns.

The mechanism asks the model — mid-investigation — to state its current
leading root-cause hypothesis and a probability that the hypothesis is
exactly right. The report is parsed from a JSON object in the reply. This is
prompting architecture layered on the stock model: no fine-tune, no logprobs,
no extra tools, no extra context about the incident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

CHECKPOINT_PROMPT = (
    "Checkpoint — do not call any tools for this message. Report your current "
    "state as a single JSON object with exactly these keys:\n"
    '{"confidence": <0.0-1.0 probability that your current hypothesis names '
    "exactly the true root-cause entity/entities, no more and no fewer>, "
    '"hypothesis": [{"name": "<entity>", "kind": "<Pod|Service|Deployment|'
    'ConfigMap|...>"}], "rationale": "<one sentence>"}\n'
    "Be calibrated: report the probability you would bet on, not optimism. "
    "If you have no hypothesis yet, use confidence 0.0 and an empty list."
)


@dataclass(frozen=True)
class ConfidenceReport:
    confidence: float
    hypothesis: tuple[dict[str, str], ...] = ()
    rationale: str = ""
    parsed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "hypothesis": list(self.hypothesis),
            "rationale": self.rationale,
            "parsed": self.parsed,
        }


UNPARSEABLE = ConfidenceReport(confidence=0.0, parsed=False)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_confidence_report(text: str) -> ConfidenceReport:
    """Extract the checkpoint JSON from a model reply, tolerantly.

    Models wrap JSON in prose or code fences; we take the first parseable
    JSON object containing a ``confidence`` key. An unparseable reply maps to
    confidence 0.0 with ``parsed=False`` — the policy treats it as "no
    signal", which biases toward continuing rather than stopping early on
    garbage.
    """
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "confidence" not in obj:
            continue
        try:
            confidence = float(obj["confidence"])
        except (TypeError, ValueError):
            continue
        confidence = min(1.0, max(0.0, confidence))
        hypothesis: list[dict[str, str]] = []
        for item in obj.get("hypothesis") or []:
            if isinstance(item, dict) and str(item.get("name", "")).strip():
                hypothesis.append(
                    {"name": str(item["name"]).strip(), "kind": str(item.get("kind", "")).strip()}
                )
            elif isinstance(item, str) and item.strip():
                hypothesis.append({"name": item.strip(), "kind": ""})
        return ConfidenceReport(
            confidence=confidence,
            hypothesis=tuple(hypothesis),
            rationale=str(obj.get("rationale", "")),
        )
    return UNPARSEABLE


def _candidates(text: str) -> list[str]:
    """JSON-object substrings to try, outermost first, then fenced blocks."""
    out: list[str] = []
    fence = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    out.extend(fence)
    match = _JSON_OBJECT.search(text)
    if match:
        out.append(match.group(0))
    # Progressively trim trailing junk from the outermost match.
    if match:
        body = match.group(0)
        for end in range(len(body), 1, -1):
            if body[end - 1] == "}":
                out.append(body[:end])
                if len(out) > 8:
                    break
    return out


@dataclass
class ConfidenceHistory:
    """Rolling view of checkpoint reports for the stopping policy."""

    reports: list[ConfidenceReport] = field(default_factory=list)

    def add(self, report: ConfidenceReport) -> None:
        self.reports.append(report)

    @property
    def latest(self) -> ConfidenceReport | None:
        return self.reports[-1] if self.reports else None

    def best(self) -> ConfidenceReport | None:
        """The highest-confidence report that carried a hypothesis."""
        with_hyp = [r for r in self.reports if r.hypothesis]
        return max(with_hyp, key=lambda r: r.confidence) if with_hyp else None

    def stalled(self, patience: int, epsilon: float = 0.05) -> bool:
        """True when the last ``patience`` checkpoints gained < epsilon."""
        if len(self.reports) < patience + 1:
            return False
        window = [r.confidence for r in self.reports[-(patience + 1):]]
        return max(window[1:]) - window[0] < epsilon
