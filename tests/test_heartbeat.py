import math
import random
from datetime import date, datetime, timedelta

from ayabada.heartbeat.baseline import SeasonalBaseline
from ayabada.heartbeat.detector import AnomalyDetector
from ayabada.heartbeat.gate import GateState, WakeGate
from ayabada.heartbeat.monitor import Heartbeat, HeartbeatConfig

START = datetime(2026, 5, 4, 0, 0)  # a Monday


def seasonal_metrics(ts: datetime, rng: random.Random) -> dict[str, float]:
    frac = (ts.hour * 60 + ts.minute) / 1440
    return {
        "traffic": max(1000 + 600 * math.sin(2 * math.pi * (frac - 0.3)) + rng.gauss(0, 25), 50),
        "error_rate": max(rng.gauss(0.01, 0.003), 0),
        "p95_latency_ms": max(rng.gauss(180, 12), 1),
    }


def feed_history(hb: Heartbeat, weeks: int, seed: int = 3) -> datetime:
    rng = random.Random(seed)
    ts = START
    for _ in range(weeks * 7 * 96):
        assert hb.ingest(ts, seasonal_metrics(ts, rng)) is None
        ts += timedelta(minutes=15)
    return ts


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------

def test_baseline_cold_start_returns_none():
    b = SeasonalBaseline(min_samples=6)
    assert b.zscore("m", START, 1.0) is None


def test_baseline_falls_back_through_hierarchy():
    b = SeasonalBaseline(min_samples=6)
    # 8 samples spread across different weekdays at the same hour: the exact
    # (dow, slot) buckets stay thin but the hour bucket fills.
    for day in range(8):
        b.observe("m", START + timedelta(days=day), 100.0 + day * 0.1)
    stats = b.stats("m", START + timedelta(days=20))
    assert stats is not None
    assert stats.level in ("hour", "global")
    assert stats.widening > 1.0  # wide bands while history is coarse


def test_baseline_tightens_with_history():
    b = SeasonalBaseline(min_samples=6)
    for week in range(8):
        b.observe("m", START + timedelta(weeks=week), 100.0)
    stats = b.stats("m", START + timedelta(weeks=9))
    assert stats.level == "dow_slot"
    assert stats.widening == 1.0


def test_zscore_flags_deviation_not_raw_value():
    b = SeasonalBaseline(min_samples=6)
    # normal is 1000 at this bucket
    for week in range(8):
        b.observe("hi", START + timedelta(weeks=week), 1000.0 + week)
        b.observe("lo", START + timedelta(weeks=week), 0.01 + week * 1e-4)
    ts = START + timedelta(weeks=9)
    z_hi, _ = b.zscore("hi", ts, 1005.0)       # tiny relative deviation
    z_lo, _ = b.zscore("lo", ts, 0.4)          # huge relative deviation, small raw value
    assert abs(z_hi) < 3
    assert abs(z_lo) > 6


# ----------------------------------------------------------------------
# Detector: corroboration
# ----------------------------------------------------------------------

def build_detector() -> AnomalyDetector:
    b = SeasonalBaseline(min_samples=6)
    for week in range(8):
        ts = START + timedelta(weeks=week)
        b.observe("error_rate", ts, 0.01)
        b.observe("latency", ts, 180.0)
        b.observe("traffic", ts, 1000.0)
    return AnomalyDetector(baseline=b, corroboration_min=2)


def test_single_metric_cannot_corroborate():
    d = build_detector()
    ts = START + timedelta(weeks=9)
    a = d.assess(ts, {"error_rate": 0.5, "latency": 181.0, "traffic": 1001.0})
    assert len(a.deviations) == 1
    assert not a.corroborated


def test_two_metrics_corroborate_and_severity_flags():
    d = build_detector()
    ts = START + timedelta(weeks=9)
    a = d.assess(ts, {"error_rate": 0.5, "latency": 900.0, "traffic": 1001.0})
    assert a.corroborated and a.severe


def test_insufficient_history_flagged():
    d = AnomalyDetector(baseline=SeasonalBaseline(min_samples=6))
    a = d.assess(START, {"error_rate": 0.5})
    assert a.insufficient_history and not a.corroborated


# ----------------------------------------------------------------------
# Gate: persistence, hysteresis, cooldown, severity
# ----------------------------------------------------------------------

def assessment(corroborated: bool, severe: bool = True, sustained: int = 0):
    from ayabada.heartbeat.detector import Deviation, IntervalAssessment

    dev = Deviation(metric="m", value=1, zscore=9.9, expected_median=0, band_level="dow_slot", samples=10)
    return IntervalAssessment(
        ts=START,
        deviations=(dev, dev) if corroborated else (),
        sustained=tuple(dev for _ in range(sustained)),
        corroborated=corroborated,
        severe=severe and corroborated,
        max_z=9.9 if corroborated else 0.1,
        insufficient_history=False,
    )


def test_gate_requires_persistence():
    g = WakeGate(persistence=3)
    assert not g.update(assessment(True)).wake
    assert not g.update(assessment(True)).wake
    d = g.update(assessment(True))
    assert d.wake and d.state is GateState.AWAKE


def test_gate_resets_on_quiet_interval():
    g = WakeGate(persistence=3)
    g.update(assessment(True))
    g.update(assessment(False, sustained=0))  # fully quiet: reset
    assert g.state is GateState.ASLEEP
    g.update(assessment(True))
    g.update(assessment(True))
    assert g.streak == 2  # streak restarted from zero, not resumed at 3


def test_gate_hysteresis_holds_streak():
    g = WakeGate(persistence=3, hold_min=2)
    g.update(assessment(True))
    g.update(assessment(True))
    # Borderline interval: below entry threshold but still elevated (2 sustained).
    d = g.update(assessment(False, sustained=2))
    assert g.state is GateState.CANDIDATE and g.streak == 2 and not d.wake
    d = g.update(assessment(True))
    assert d.wake  # streak resumed at 3, not restarted


def test_gate_severity_gating_blocks_mild_anomalies():
    g = WakeGate(persistence=2, require_severe=True)
    g.update(assessment(True, severe=False))
    d = g.update(assessment(True, severe=False))
    assert not d.wake
    assert "severity" in d.reason


def test_gate_cooldown_suppresses_rewake():
    g = WakeGate(persistence=1, cooldown_intervals=3)
    assert g.update(assessment(True)).wake
    g.update(assessment(False, sustained=0))  # episode ends -> cooldown
    assert g.state is GateState.COOLDOWN
    d = g.update(assessment(True))  # recurrence during cooldown
    assert not d.wake and g.state is GateState.AWAKE


# ----------------------------------------------------------------------
# Monitor end-to-end
# ----------------------------------------------------------------------

def test_no_false_wakes_on_clean_weeks():
    hb = Heartbeat(HeartbeatConfig())
    feed_history(hb, weeks=3)  # asserts no wake inside


def test_wakes_once_on_injected_incident_with_packaged_snapshot():
    hb = Heartbeat(HeartbeatConfig())
    ts = feed_history(hb, weeks=3)
    rng = random.Random(5)
    snapshots = []
    for _ in range(12):
        obs = seasonal_metrics(ts, rng)
        obs["error_rate"] = 0.4 + rng.gauss(0, 0.02)
        obs["p95_latency_ms"] = 800 + rng.gauss(0, 40)
        snap = hb.ingest(ts, obs)
        if snap:
            snapshots.append(snap)
        ts += timedelta(minutes=15)
    assert len(snapshots) == 1  # exactly one wake, no flapping
    snap = snapshots[0]
    assert snap.source == "heartbeat"
    assert {a.labels["metric"] for a in snap.alerts} == {"error_rate", "p95_latency_ms"}
    assert snap.metrics  # telemetry window packaged
    assert "persisted" in snap.extra["gate_reason"]


def test_single_metric_spike_never_wakes():
    hb = Heartbeat(HeartbeatConfig())
    ts = feed_history(hb, weeks=3)
    rng = random.Random(6)
    for _ in range(12):
        obs = seasonal_metrics(ts, rng)
        obs["error_rate"] = 0.4  # only one metric deviates
        assert hb.ingest(ts, obs) is None
        ts += timedelta(minutes=15)


def test_special_day_suppresses_wake():
    incident_day = (START + timedelta(weeks=3)).date()
    config = HeartbeatConfig(special_days={incident_day}, special_day_multiplier=25.0)
    hb = Heartbeat(config)
    ts = feed_history(hb, weeks=3)
    assert ts.date() == incident_day
    rng = random.Random(7)
    for _ in range(12):
        obs = seasonal_metrics(ts, rng)
        # Black-Friday-like traffic surge: high but expected on a flagged day.
        obs["traffic"] = obs["traffic"] * 3
        obs["p95_latency_ms"] = obs["p95_latency_ms"] * 1.5
        assert hb.ingest(ts, obs) is None
        ts += timedelta(minutes=15)
