"""The wake gate: killing false wakes is the hard part, not detection.

State machine over per-interval assessments:

- **ASLEEP** → corroborated anomaly starts a **CANDIDATE** streak.
- **CANDIDATE** → the anomaly must persist ``persistence`` consecutive
  intervals (with at least one severe deviation, if required) before the
  agent wakes. Hysteresis: a borderline interval (still above ``z_exit``)
  holds the streak instead of resetting it, so noise at the threshold
  doesn't restart the count; a genuinely quiet interval resets to ASLEEP.
- **AWAKE** → the episode continues while deviations sustain; no new wakes.
- **COOLDOWN** → after an episode ends, re-waking is suppressed for
  ``cooldown_intervals``; a recurrence during cooldown re-enters AWAKE
  silently (it's the same incident, not a new one).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ayabada.heartbeat.detector import IntervalAssessment


class GateState(Enum):
    ASLEEP = "asleep"
    CANDIDATE = "candidate"
    AWAKE = "awake"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class GateDecision:
    wake: bool
    state: GateState
    streak: int
    reason: str


@dataclass
class WakeGate:
    persistence: int = 3
    cooldown_intervals: int = 8
    require_severe: bool = True
    # Hysteresis corroboration: how many metrics must stay above z_exit for
    # an interval to count as "still elevated" rather than quiet.
    hold_min: int = 2

    state: GateState = GateState.ASLEEP
    streak: int = 0
    severe_seen: bool = False
    cooldown_remaining: int = 0

    def update(self, assessment: IntervalAssessment) -> GateDecision:
        corroborated = assessment.corroborated
        # Hysteresis view: enough metrics still elevated above the exit band.
        holding = len(assessment.sustained) >= self.hold_min

        if self.state in (GateState.ASLEEP, GateState.CANDIDATE):
            if corroborated:
                was_asleep = self.state is GateState.ASLEEP
                self.state = GateState.CANDIDATE
                self.streak += 1
                self.severe_seen = self.severe_seen or assessment.severe
                if self.streak >= self.persistence and (
                    self.severe_seen or not self.require_severe
                ):
                    self.state = GateState.AWAKE
                    return self._decision(
                        True,
                        f"anomaly persisted {self.streak} interval(s)"
                        + (" with severe deviation" if self.severe_seen else ""),
                    )
                if self.streak >= self.persistence:
                    return self._decision(
                        False, "persistent but below severity gate — not waking"
                    )
                return self._decision(
                    False,
                    "anomaly candidate started" if was_asleep else "candidate streak building",
                )
            if self.state is GateState.CANDIDATE and holding:
                # Borderline interval: hold the streak, don't reset.
                return self._decision(False, "holding streak through borderline interval")
            if self.state is GateState.CANDIDATE:
                self.state = GateState.ASLEEP
                self.streak = 0
                self.severe_seen = False
                return self._decision(False, "anomaly did not persist; reset")
            return self._decision(False, "quiet")

        if self.state is GateState.AWAKE:
            if corroborated or holding:
                return self._decision(False, "episode ongoing")
            self.state = GateState.COOLDOWN
            self.cooldown_remaining = self.cooldown_intervals
            return self._decision(False, "episode ended; entering cooldown")

        # COOLDOWN
        if corroborated:
            self.state = GateState.AWAKE
            return self._decision(False, "recurrence during cooldown; same episode")
        self.cooldown_remaining -= 1
        if self.cooldown_remaining <= 0:
            self.state = GateState.ASLEEP
            self.streak = 0
            self.severe_seen = False
            return self._decision(False, "cooldown complete")
        return self._decision(False, "cooling down")

    def _decision(self, wake: bool, reason: str) -> GateDecision:
        return GateDecision(wake=wake, state=self.state, streak=self.streak, reason=reason)
