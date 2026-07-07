"""The heartbeat: compose baseline + detector + gate; wake with a snapshot.

``Heartbeat.ingest(ts, observations)`` is the entire production loop body:
feed it one dict of metric readings per interval (traffic, request rate,
error rate, p95 latency, saturation, ...) and it returns an
:class:`~ayabada.snapshot.IncidentSnapshot` on the single interval where the
gate decides to wake the brain, and ``None`` otherwise.

The snapshot it emits has the same shape the benchmark hands the brain —
synthetic alerts for the corroborating deviations plus the recent metric
window — so the brain cannot tell which side of the system woke it. This is
production-only plumbing and never appears inside a scored benchmark run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from ayabada.heartbeat.baseline import SeasonalBaseline
from ayabada.heartbeat.calendars import SpecialDayCalendar
from ayabada.heartbeat.detector import AnomalyDetector, IntervalAssessment
from ayabada.heartbeat.gate import GateDecision, GateState, WakeGate
from ayabada.snapshot import Alert, IncidentSnapshot, MetricSeries


@dataclass
class HeartbeatConfig:
    min_samples: int = 6
    corroboration_min: int = 2
    z_enter: float = 3.0
    z_severe: float = 6.0
    z_exit: float = 2.0
    persistence: int = 3
    cooldown_intervals: int = 8
    require_severe: bool = True
    window_intervals: int = 16   # how much recent telemetry a wake packages
    special_day_multiplier: float = 2.0
    special_days: set = field(default_factory=set)


class Heartbeat:
    def __init__(self, config: HeartbeatConfig | None = None) -> None:
        self.config = config or HeartbeatConfig()
        calendar = SpecialDayCalendar(
            dates=set(self.config.special_days),
            band_multiplier=self.config.special_day_multiplier,
        )
        self.baseline = SeasonalBaseline(min_samples=self.config.min_samples)
        self.detector = AnomalyDetector(
            baseline=self.baseline,
            corroboration_min=self.config.corroboration_min,
            z_enter=self.config.z_enter,
            z_severe=self.config.z_severe,
            z_exit=self.config.z_exit,
            calendar=calendar,
        )
        self.gate = WakeGate(
            persistence=self.config.persistence,
            cooldown_intervals=self.config.cooldown_intervals,
            require_severe=self.config.require_severe,
            hold_min=self.config.corroboration_min,
        )
        self._window: deque[tuple[datetime, dict[str, float]]] = deque(
            maxlen=self.config.window_intervals
        )
        self._wake_count = 0
        self.last_assessment: IntervalAssessment | None = None
        self.last_decision: GateDecision | None = None

    @property
    def state(self) -> GateState:
        return self.gate.state

    def ingest(self, ts: datetime, observations: dict[str, float]) -> IncidentSnapshot | None:
        """One interval: assess against history, gate, then record.

        Assessment happens *before* the observation is recorded so a value is
        never judged against a bucket it has already shifted.
        """
        assessment = self.detector.assess(ts, observations)
        decision = self.gate.update(assessment)
        self.last_assessment = assessment
        self.last_decision = decision

        for metric, value in observations.items():
            self.baseline.observe(metric, ts, value)
        self._window.append((ts, dict(observations)))

        if not decision.wake:
            return None
        self._wake_count += 1
        return self._package(ts, assessment, decision)

    def _package(
        self, ts: datetime, assessment: IntervalAssessment, decision: GateDecision
    ) -> IncidentSnapshot:
        """Bundle the anomalous window into the shared snapshot shape."""
        alerts = tuple(
            Alert(
                name=f"HeartbeatAnomaly:{dev.metric}",
                severity="critical" if abs(dev.zscore) >= self.detector.z_severe else "warning",
                labels={"metric": dev.metric, "band_level": dev.band_level},
                description=(
                    f"{dev.metric}={dev.value:g} deviates z={dev.zscore:+.1f} from the "
                    f"seasonal norm (median {dev.expected_median:g}, "
                    f"{dev.samples} samples, {dev.band_level} bucket)"
                ),
                active_at=ts.isoformat(),
                value=dev.zscore,
            )
            for dev in assessment.deviations
        )

        by_metric: dict[str, list[tuple[float, float]]] = {}
        for w_ts, obs in self._window:
            for metric, value in obs.items():
                by_metric.setdefault(metric, []).append((w_ts.timestamp(), value))
        metrics = tuple(
            MetricSeries(metric=m, entity=m, samples=tuple(samples))
            for m, samples in sorted(by_metric.items())
        )

        window_start = self._window[0][0] if self._window else ts
        deviating = ", ".join(
            f"{d.metric} (z={d.zscore:+.1f})" for d in assessment.deviations
        )
        return IncidentSnapshot(
            id=f"heartbeat-{ts.strftime('%Y%m%dT%H%M')}-{self._wake_count}",
            source="heartbeat",
            description=(
                "Live anomaly detected by the metric heartbeat: "
                f"{decision.reason}. Deviating metrics: {deviating}. "
                "Investigate the packaged telemetry window and identify the "
                "root cause."
            ),
            window_start=window_start.isoformat(),
            window_end=ts.isoformat(),
            alerts=alerts,
            metrics=metrics,
            extra={"gate_reason": decision.reason, "streak": decision.streak},
        )
