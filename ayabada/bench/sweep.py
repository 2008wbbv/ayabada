"""Ablation sweep: the confidence mechanism at many operating points.

A sweep is just a head-to-head with more arms — one baseline plus one
confidence arm per policy in the grid. Every arm goes through the same
parity check (same model, same tools; only ``architecture_params`` differ),
so a sweep result is a set of directly comparable operating points:
score vs trajectory length vs escalation rate as the thresholds move.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

from ayabada.bench.runner import ArmConfig, HeadToHeadResult
from ayabada.bench.scoring import aggregate
from ayabada.brain.loop import DEFAULT_MAX_TURNS
from ayabada.brain.policy import StoppingPolicy


def sweep_arms(
    stop_thresholds: Iterable[float] = (0.6, 0.8, 0.9),
    escalate_thresholds: Iterable[float] = (0.35,),
    patiences: Iterable[int] = (3,),
    checkpoint_everys: Iterable[int] = (1,),
    max_turns: int = DEFAULT_MAX_TURNS,
    include_baseline: bool = True,
) -> tuple[ArmConfig, ...]:
    """Build the arm list for a policy grid; names encode the operating point."""
    arms: list[ArmConfig] = []
    if include_baseline:
        arms.append(ArmConfig(name="baseline", architecture="baseline", max_turns=max_turns))
    for stop, esc, patience, every in product(
        stop_thresholds, escalate_thresholds, patiences, checkpoint_everys
    ):
        if esc >= stop:
            continue  # nonsensical operating point: would escalate above the stop bar
        arms.append(
            ArmConfig(
                name=f"conf_s{stop:g}_e{esc:g}_p{patience}_c{every}",
                architecture="confidence",
                max_turns=max_turns,
                policy=StoppingPolicy(
                    stop_threshold=stop,
                    escalate_threshold=esc,
                    patience=patience,
                    checkpoint_every=every,
                    max_turns=max_turns,
                ),
            )
        )
    if len(arms) <= int(include_baseline):
        raise ValueError("sweep grid produced no confidence arms")
    return tuple(arms)


def sweep_table(result: HeadToHeadResult) -> list[dict[str, Any]]:
    """One row per arm, sorted best score first (ties: fewer turns wins)."""
    rows = []
    for arm_name, scores in result.task_scores.items():
        agg = aggregate(scores)
        spec = result.arms.get(arm_name, {})
        rows.append(
            {
                "arm": arm_name,
                "architecture": spec.get("architecture", "?"),
                **{k: round(v, 3) for k, v in agg.items()},
            }
        )
    return sorted(rows, key=lambda r: (-r.get("mean_score", 0.0), r.get("mean_turns", 0.0)))


def sweep_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Arm | Score | P | R | Turns | Tool calls | Escalation | Perfect |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['mean_score']:.2f} | {row['mean_precision']:.2f} "
            f"| {row['mean_recall']:.2f} | {row['mean_turns']:.1f} "
            f"| {row['mean_tool_calls']:.1f} | {row['escalation_rate']:.0%} "
            f"| {row['perfect_rate']:.0%} |"
        )
    return "\n".join(lines)
