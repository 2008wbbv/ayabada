"""Deseasonalize, then combine: per-interval anomaly assessment.

Each metric is first converted to its deviation from its own seasonal bucket
(robust z-score); only then are metrics combined. An interval is *anomalous*
only when at least ``corroboration_min`` metrics corroborate (each beyond
``z_enter``), and *severe* when at least one deviation is big enough
(``z_severe``) to justify waking the full agent. A single twitchy metric can
never wake anything by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ayabada.heartbeat.baseline import BucketStats, SeasonalBaseline
from ayabada.heartbeat.calendars import SpecialDayCalendar


@dataclass(frozen=True)
class Deviation:
    metric: str
    value: float
    zscore: float
    expected_median: float
    band_level: str
    samples: int


@dataclass(frozen=True)
class IntervalAssessment:
    ts: datetime
    deviations: tuple[Deviation, ...]   # metrics beyond z_enter this interval
    sustained: tuple[Deviation, ...]    # metrics beyond z_exit (hysteresis band)
    corroborated: bool                  # >= corroboration_min beyond z_enter
    severe: bool                        # >= 1 beyond z_severe
    max_z: float
    insufficient_history: bool          # no metric had enough history to judge


@dataclass
class AnomalyDetector:
    baseline: SeasonalBaseline
    corroboration_min: int = 2
    z_enter: float = 3.0     # deviation that counts toward corroboration
    z_severe: float = 6.0    # at least one metric must deviate this hard
    z_exit: float = 2.0      # hysteresis: an episode persists above this
    calendar: SpecialDayCalendar = field(default_factory=SpecialDayCalendar)

    def assess(self, ts: datetime, observations: dict[str, float]) -> IntervalAssessment:
        """Judge one interval. Does NOT update the baseline — the monitor
        records observations after assessment so a value can't shift the very
        bucket it is being judged against."""
        widen = self.calendar.multiplier_for(ts)
        deviations: list[Deviation] = []
        sustained: list[Deviation] = []
        max_z = 0.0
        judged = 0

        for metric, value in observations.items():
            scored = self.baseline.zscore(metric, ts, value)
            if scored is None:
                continue
            judged += 1
            z, stats = scored
            z_eff = z / widen
            abs_z = abs(z_eff)
            max_z = max(max_z, abs_z)
            dev = Deviation(
                metric=metric,
                value=value,
                zscore=round(z_eff, 3),
                expected_median=stats.median,
                band_level=stats.level,
                samples=stats.n,
            )
            if abs_z >= self.z_enter:
                deviations.append(dev)
            if abs_z >= self.z_exit:
                sustained.append(dev)

        return IntervalAssessment(
            ts=ts,
            deviations=tuple(sorted(deviations, key=lambda d: -abs(d.zscore))),
            sustained=tuple(sorted(sustained, key=lambda d: -abs(d.zscore))),
            corroborated=len(deviations) >= self.corroboration_min,
            severe=any(abs(d.zscore) >= self.z_severe for d in deviations),
            max_z=max_z,
            insufficient_history=judged == 0,
        )
