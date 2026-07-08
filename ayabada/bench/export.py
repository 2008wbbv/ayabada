"""Export run predictions for external validation / leaderboard submission.

The official ITBench evaluation is a hosted service — you submit your
agent's outputs and they score them. This module produces that bundle from
a runner ``records.jsonl``: per scenario, the raw predicted entities, the
canonical group ids our scorer resolved them to, and our local score. When
the official numbers come back, diffing them against ``local_score`` per
scenario is the scoring cross-validation (any disagreement means our
reimplementation of recall-gated precision diverges and must be fixed
before claiming deltas).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ayabada.bench.ground_truth import GroundTruth


def export_predictions(
    records_path: Path,
    ground_truths: dict[str, GroundTruth],
    arm: str = "confidence",
) -> dict[str, Any]:
    submissions: list[dict[str, Any]] = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("arm") != arm:
            continue
        scenario_id = record.get("scenario_id", "")
        gt = ground_truths.get(scenario_id)
        run = record.get("run") or {}
        predictions = run.get("predictions") or []
        resolved = []
        for pred in predictions:
            canonical = (
                gt.resolve(pred.get("name", ""), pred.get("kind") or None) if gt else None
            )
            resolved.append({**pred, "canonical_group": canonical})
        submissions.append(
            {
                "scenario_id": scenario_id,
                "predicted_entities": resolved,
                "escalated": run.get("escalated", False),
                "summary": run.get("summary", ""),
                "num_turns": run.get("num_turns", 0),
                "local_score": (record.get("score") or {}).get("score"),
                "local_precision": (record.get("score") or {}).get("precision"),
                "local_recall": (record.get("score") or {}).get("recall"),
            }
        )
    return {"arm": arm, "n_scenarios": len(submissions), "submissions": submissions}
