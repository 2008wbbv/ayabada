"""Scoring: recall-gated precision over root-cause entities.

The measured weakness this project targets is *not knowing when to stop*:
frontier models keep investigating, name symptom entities alongside the true
root cause, and long trajectories score worse rather than better. The metric
therefore punishes exactly that — every predicted entity beyond the true
root-cause set lowers precision, and the headline score gates precision on
recall so partial answers can't be gamed by shotgunning entity names.

Definitions per task:

- Predictions are mapped to canonical entity-group ids via the ground truth's
  regex groups + alias sets. A prediction that maps to no known group is a
  false positive (the agent named something that isn't even in the incident's
  entity universe).
- ``precision = |TP| / |predicted|``, ``recall = |TP| / |positives|``.
- ``score = precision if recall >= recall_gate else 0.0`` (default gate 1.0:
  you must find *all* root-cause entities before precision counts).

An escalation (the agent declining to answer) scores 0 for the task but is
tracked separately — the whole point of calibrated escalation is that a
0-scored honest "I don't know" is operationally cheaper than a confidently
wrong answer, and we want that visible in the results, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from ayabada.bench.ground_truth import GroundTruth


@dataclass(frozen=True)
class Prediction:
    """One entity the agent blames, e.g. ('flagd-config', 'ConfigMap')."""

    name: str
    kind: str = ""


@dataclass
class TaskScore:
    scenario_id: str
    precision: float
    recall: float
    score: float
    true_positives: list[str]
    false_positives: list[str]
    false_negatives: list[str]
    escalated: bool = False
    num_turns: int = 0
    num_tool_calls: int = 0
    stop_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "score": self.score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "escalated": self.escalated,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "stop_reason": self.stop_reason,
            **({"meta": self.meta} if self.meta else {}),
        }


def score_task(
    ground_truth: GroundTruth,
    predictions: list[Prediction],
    *,
    escalated: bool = False,
    num_turns: int = 0,
    num_tool_calls: int = 0,
    stop_reason: str = "",
    recall_gate: float = 1.0,
) -> TaskScore:
    positives = ground_truth.root_cause_ids()

    resolved: set[str] = set()
    unresolved: list[str] = []
    for pred in predictions:
        canonical = ground_truth.resolve(pred.name, pred.kind or None)
        if canonical is None:
            unresolved.append(pred.name)
        else:
            resolved.add(canonical)

    tp = sorted(resolved & positives)
    fp = sorted(resolved - positives) + sorted(set(unresolved))
    fn = sorted(positives - resolved)

    n_predicted = len(resolved) + len(set(unresolved))
    precision = len(tp) / n_predicted if n_predicted else 0.0
    recall = len(tp) / len(positives) if positives else 0.0
    score = precision if recall >= recall_gate else 0.0

    return TaskScore(
        scenario_id=ground_truth.scenario_id,
        precision=precision,
        recall=recall,
        score=score,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        escalated=escalated,
        num_turns=num_turns,
        num_tool_calls=num_tool_calls,
        stop_reason=stop_reason,
    )


def aggregate(scores: list[TaskScore]) -> dict[str, float]:
    """Arm-level summary across tasks."""
    if not scores:
        return {}
    return {
        "n_tasks": len(scores),
        "mean_score": mean(s.score for s in scores),
        "mean_precision": mean(s.precision for s in scores),
        "mean_recall": mean(s.recall for s in scores),
        "mean_f1": mean(s.f1 for s in scores),
        "mean_turns": mean(s.num_turns for s in scores),
        "mean_tool_calls": mean(s.num_tool_calls for s in scores),
        "escalation_rate": mean(1.0 if s.escalated else 0.0 for s in scores),
        "perfect_rate": mean(1.0 if s.score == 1.0 else 0.0 for s in scores),
    }
