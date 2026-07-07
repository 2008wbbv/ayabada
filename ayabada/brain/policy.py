"""The calibrated stopping/escalation policy — the research contribution.

Given the rolling history of confidence checkpoints, decide after each turn
whether to keep investigating, stop and submit the current hypothesis, or
halt and escalate to a human. The policy is deliberately simple and fully
inspectable; every threshold is an ablatable parameter.

Decision rules, in order:

1. **Stop**: latest confidence ≥ ``stop_threshold`` and a hypothesis exists
   (after ``min_turns`` — one lucky early guess shouldn't end the run).
2. **Escalate on stall**: confidence has improved < ``epsilon`` over the last
   ``patience`` checkpoints while staying below ``escalate_threshold`` —
   more investigation is not producing signal, so stop burning turns and
   hand off.
3. **Budget exhausted** (at ``max_turns``): submit the best hypothesis seen
   if its confidence ≥ ``escalate_threshold``, otherwise escalate. The
   baseline in the same situation submits whatever it has; the ablation
   difference is exactly this honesty gate.
4. Otherwise: continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ayabada.brain.confidence import ConfidenceHistory


class Decision(Enum):
    CONTINUE = "continue"
    STOP = "stop"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class StoppingPolicy:
    stop_threshold: float = 0.8
    escalate_threshold: float = 0.35
    patience: int = 3
    epsilon: float = 0.05
    min_turns: int = 2
    checkpoint_every: int = 1
    max_turns: int = 30

    def decide(self, history: ConfidenceHistory, turn: int) -> tuple[Decision, str]:
        latest = history.latest
        if latest is None:
            return Decision.CONTINUE, "no checkpoints yet"

        if (
            turn >= self.min_turns
            and latest.parsed
            and latest.hypothesis
            and latest.confidence >= self.stop_threshold
        ):
            return (
                Decision.STOP,
                f"confidence {latest.confidence:.2f} >= stop threshold {self.stop_threshold}",
            )

        if (
            history.stalled(self.patience, self.epsilon)
            and latest.confidence < self.escalate_threshold
        ):
            return (
                Decision.ESCALATE,
                f"stalled below escalate threshold for {self.patience} checkpoints "
                f"(confidence {latest.confidence:.2f})",
            )

        return Decision.CONTINUE, "keep investigating"

    def decide_at_budget(self, history: ConfidenceHistory) -> tuple[Decision, str]:
        """Turn budget exhausted: submit best hypothesis or escalate honestly."""
        best = history.best()
        if best is not None and best.confidence >= self.escalate_threshold:
            return (
                Decision.STOP,
                f"budget exhausted; submitting best hypothesis at {best.confidence:.2f}",
            )
        conf = best.confidence if best else 0.0
        return (
            Decision.ESCALATE,
            f"budget exhausted with confidence {conf:.2f} < {self.escalate_threshold}",
        )

    def to_dict(self) -> dict:
        return {
            "stop_threshold": self.stop_threshold,
            "escalate_threshold": self.escalate_threshold,
            "patience": self.patience,
            "epsilon": self.epsilon,
            "min_turns": self.min_turns,
            "checkpoint_every": self.checkpoint_every,
            "max_turns": self.max_turns,
        }
