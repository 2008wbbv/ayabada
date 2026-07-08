"""Calibration measurement: is the reported confidence worth anything?

The mechanism's claim is *calibrated* stopping — the model's stated
probability that its hypothesis names exactly the true root cause should
track how often that hypothesis actually is exactly right. This module turns
a head-to-head ``records.jsonl`` into reliability-diagram data:

- Every checkpoint in every confidence trace contributes one
  ``(confidence, correct)`` pair, where *correct* means the checkpoint's
  hypothesis would have scored 1.0 (all root-cause entities, nothing else)
  had the run stopped right there.
- Pairs are binned by confidence; each bin reports its size, mean stated
  confidence and empirical accuracy. The summary statistic is ECE
  (expected calibration error): the bin-size-weighted mean |accuracy − confidence|.

Both arms' traces can be measured (the baseline has none), and the same
data drives threshold selection: the stop threshold should sit where the
empirical accuracy curve crosses the precision you need.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ayabada.bench.ground_truth import GroundTruth
from ayabada.bench.scoring import Prediction, score_task


@dataclass(frozen=True)
class CalibrationPoint:
    confidence: float
    correct: bool
    scenario_id: str = ""
    arm: str = ""
    turn: int = 0


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    points: list[CalibrationPoint] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def mean_confidence(self) -> float:
        return sum(p.confidence for p in self.points) / self.n if self.n else 0.0

    @property
    def accuracy(self) -> float:
        return sum(1 for p in self.points if p.correct) / self.n if self.n else 0.0


@dataclass
class CalibrationReport:
    bins: list[CalibrationBin]
    n_points: int
    ece: float
    arm: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_points": self.n_points,
            "ece": round(self.ece, 4),
            "bins": [
                {
                    "range": [b.lo, b.hi],
                    "n": b.n,
                    "mean_confidence": round(b.mean_confidence, 3),
                    "accuracy": round(b.accuracy, 3),
                }
                for b in self.bins
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"### Calibration — {self.arm or 'all arms'} "
            f"({self.n_points} checkpoints, ECE {self.ece:.3f})",
            "",
            "| Confidence bin | n | Stated (mean) | Empirical accuracy | Gap |",
            "|---|---|---|---|---|",
        ]
        for b in self.bins:
            if b.n == 0:
                continue
            gap = b.accuracy - b.mean_confidence
            lines.append(
                f"| {b.lo:.1f}–{b.hi:.1f} | {b.n} | {b.mean_confidence:.2f} "
                f"| {b.accuracy:.2f} | {gap:+.2f} |"
            )
        return "\n".join(lines)


def hypothesis_correct(ground_truth: GroundTruth, hypothesis: list[dict[str, str]]) -> bool:
    """Would this hypothesis have scored 1.0 (perfect recall-gated precision)?"""
    if not hypothesis:
        return False
    predictions = [
        Prediction(name=h.get("name", ""), kind=h.get("kind", "")) for h in hypothesis
    ]
    return score_task(ground_truth, predictions).score == 1.0


def collect_points(
    records: Iterable[dict[str, Any]],
    ground_truths: dict[str, GroundTruth],
) -> list[CalibrationPoint]:
    """One point per parsed checkpoint across every record with a trace."""
    points: list[CalibrationPoint] = []
    for record in records:
        gt = ground_truths.get(record.get("scenario_id", ""))
        if gt is None:
            continue
        trace = (record.get("run") or {}).get("confidence_trace") or []
        for checkpoint in trace:
            if not checkpoint.get("parsed", True):
                continue
            points.append(
                CalibrationPoint(
                    confidence=float(checkpoint.get("confidence", 0.0)),
                    correct=hypothesis_correct(gt, checkpoint.get("hypothesis") or []),
                    scenario_id=record.get("scenario_id", ""),
                    arm=record.get("arm", ""),
                    turn=int(checkpoint.get("turn", 0)),
                )
            )
    return points


def bin_points(points: list[CalibrationPoint], n_bins: int = 10, arm: str = "") -> CalibrationReport:
    bins = [
        CalibrationBin(lo=i / n_bins, hi=(i + 1) / n_bins) for i in range(n_bins)
    ]
    for point in points:
        index = min(int(point.confidence * n_bins), n_bins - 1)
        bins[index].points.append(point)
    total = len(points)
    ece = (
        sum(b.n * abs(b.accuracy - b.mean_confidence) for b in bins) / total
        if total
        else 0.0
    )
    return CalibrationReport(bins=bins, n_points=total, ece=ece, arm=arm)


def calibration_from_records(
    records_path: Path,
    ground_truths: dict[str, GroundTruth],
    n_bins: int = 10,
) -> dict[str, CalibrationReport]:
    """Per-arm calibration reports from a runner records.jsonl."""
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    points = collect_points(records, ground_truths)
    reports: dict[str, CalibrationReport] = {}
    for arm in sorted({p.arm for p in points}):
        arm_points = [p for p in points if p.arm == arm]
        reports[arm] = bin_points(arm_points, n_bins=n_bins, arm=arm)
    return reports
