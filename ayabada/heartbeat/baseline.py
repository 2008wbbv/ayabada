"""Seasonal baseline: "the normal range for this day and time".

History is bucketed by (day-of-week, 15-minute slot). Per bucket we keep a
bounded sample of past values and compute robust statistics (median + MAD),
so a handful of outliers — including the incidents themselves — cannot drag
the notion of normal around.

Cold start is handled with a bucket hierarchy: exact (dow, slot) → (dow,
hour) → (hour across all days) → global. ``stats()`` uses the most specific
level with enough samples and reports which level answered; less specific
levels get their bands widened by the detector (that's the "wide bands until
2–3 weeks of history" behavior — as history accumulates, queries naturally
tighten to the exact bucket).

Raw values are never thresholded anywhere in the heartbeat: every metric is
converted to a deviation from its own bucket via :meth:`SeasonalBaseline.zscore`
first, and only those robust z-scores are compared across metrics.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from statistics import median

# Consistency constant: MAD * 1.4826 estimates the standard deviation for
# normal data, so robust z-scores are comparable to ordinary z-scores.
MAD_TO_SIGMA = 1.4826

# Band widening per fallback level: exact bucket, dow+hour, hour, global.
LEVEL_WIDENING = (1.0, 1.5, 2.0, 3.0)
LEVEL_NAMES = ("dow_slot", "dow_hour", "hour", "global")


def bucket_keys(ts: datetime) -> tuple[tuple, ...]:
    """The hierarchy of bucket keys for a timestamp, most specific first."""
    slot = (ts.hour * 60 + ts.minute) // 15
    return (
        ("dow_slot", ts.weekday(), slot),
        ("dow_hour", ts.weekday(), ts.hour),
        ("hour", ts.hour),
        ("global",),
    )


@dataclass(frozen=True)
class BucketStats:
    median: float
    mad: float
    n: int
    level: str          # which hierarchy level answered
    widening: float     # band multiplier for that level


class SeasonalBaseline:
    """Bucketed robust history for many metrics."""

    def __init__(self, min_samples: int = 6, max_per_bucket: int = 400) -> None:
        self.min_samples = min_samples
        self.max_per_bucket = max_per_bucket
        # (metric, *bucket_key) -> deque of values
        self._buckets: dict[tuple, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_per_bucket)
        )

    def observe(self, metric: str, ts: datetime, value: float) -> None:
        """Record a sample into every level of the hierarchy.

        Anomalous values are recorded too: median/MAD is robust to the
        minority of samples an incident contributes, and always-recording
        means genuine level shifts (a deploy that doubles traffic) are
        learned instead of alarming forever.
        """
        for key in bucket_keys(ts):
            self._buckets[(metric, *key)].append(value)

    def stats(self, metric: str, ts: datetime) -> BucketStats | None:
        """Robust stats from the most specific bucket with enough history."""
        for level, key in enumerate(bucket_keys(ts)):
            values = self._buckets.get((metric, *key))
            if values is not None and len(values) >= self.min_samples:
                med = median(values)
                mad = median(abs(v - med) for v in values)
                return BucketStats(
                    median=med,
                    mad=mad,
                    n=len(values),
                    level=LEVEL_NAMES[level],
                    widening=LEVEL_WIDENING[level],
                )
        return None

    def zscore(self, metric: str, ts: datetime, value: float) -> tuple[float, BucketStats] | None:
        """Deseasonalized robust z-score, or None with insufficient history.

        The z-score is already deflated by the fallback level's widening
        factor, so callers compare every metric against the same thresholds.
        """
        stats = self.stats(metric, ts)
        if stats is None:
            return None
        # Floor the scale so a perfectly flat history (MAD 0) doesn't turn
        # microscopic wiggles into infinite z-scores.
        scale = max(stats.mad * MAD_TO_SIGMA, abs(stats.median) * 0.01, 1e-9)
        z = (value - stats.median) / (scale * stats.widening)
        return z, stats
