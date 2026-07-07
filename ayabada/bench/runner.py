"""Head-to-head benchmark runner: baseline vs confidence arm, same everything.

The runner is where the experiment's one rule becomes mechanical:

- Both arms are built from a **single** ``llm_factory`` — they cannot receive
  different models by construction.
- Both arms get tools from the **same** ``SnapshotToolbox`` class — identical
  schemas by construction.
- :func:`ayabada.bench.parity.assert_parity` hashes model, model params, tool
  schemas and task instructions before any task runs, and the shared
  fingerprint is written into the results header. Only ``architecture`` and
  ``architecture_params`` may differ between arms.

Results are JSONL (one record per task × arm) plus an aggregate summary,
so ablations (mechanism on vs off) are a diff of two arm summaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ayabada.bench.itbench import TASK_INSTRUCTIONS, Scenario, load_snapshot
from ayabada.bench.parity import ArmSpec, assert_parity
from ayabada.bench.scoring import Prediction, TaskScore, aggregate, score_task
from ayabada.brain.agent import run_confidence_loop
from ayabada.brain.loop import DEFAULT_MAX_TURNS, SYSTEM_PROMPT, RunResult, run_tool_loop
from ayabada.brain.policy import StoppingPolicy
from ayabada.brain.tools import SnapshotToolbox
from ayabada.llm.base import LLMClient
from ayabada.snapshot import IncidentSnapshot


@dataclass(frozen=True)
class ArmConfig:
    """One arm of the experiment. Architecture is the only allowed variable."""

    name: str
    architecture: str  # "baseline" | "confidence"
    max_turns: int = DEFAULT_MAX_TURNS
    policy: StoppingPolicy | None = None

    def run(self, llm: LLMClient, toolbox: SnapshotToolbox) -> RunResult:
        if self.architecture == "baseline":
            return run_tool_loop(llm, toolbox, max_turns=self.max_turns)
        if self.architecture == "confidence":
            policy = self.policy or StoppingPolicy(max_turns=self.max_turns)
            return run_confidence_loop(llm, toolbox, policy=policy)
        raise ValueError(f"unknown architecture: {self.architecture}")

    def architecture_params(self) -> dict[str, Any]:
        if self.architecture == "confidence":
            policy = self.policy or StoppingPolicy(max_turns=self.max_turns)
            return policy.to_dict()
        return {"max_turns": self.max_turns}


DEFAULT_ARMS = (
    ArmConfig(name="baseline", architecture="baseline"),
    ArmConfig(name="confidence", architecture="confidence"),
)


@dataclass
class HeadToHeadResult:
    environment_fingerprint: str
    model: str
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_scores: dict[str, list[TaskScore]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "environment_fingerprint": self.environment_fingerprint,
            "model": self.model,
            "arms": {
                name: {**spec, "aggregate": aggregate(self.task_scores.get(name, []))}
                for name, spec in self.arms.items()
            },
        }


def build_arm_specs(
    llm_factory: Callable[[], LLMClient],
    arms: tuple[ArmConfig, ...],
) -> tuple[list[ArmSpec], LLMClient]:
    """Probe one client from the shared factory and declare every arm."""
    probe = llm_factory()
    tool_schemas = tuple(SnapshotToolbox.schemas())
    task_instructions = SYSTEM_PROMPT + "\n---\n" + TASK_INSTRUCTIONS
    specs = [
        ArmSpec(
            name=arm.name,
            model=probe.model,
            model_params=dict(probe.model_params),
            tool_schemas=tool_schemas,
            task_instructions=task_instructions,
            architecture=arm.architecture,
            architecture_params=arm.architecture_params(),
        )
        for arm in arms
    ]
    return specs, probe


def run_head_to_head(
    scenarios: list[Scenario],
    llm_factory: Callable[[], LLMClient],
    arms: tuple[ArmConfig, ...] = DEFAULT_ARMS,
    *,
    cache_dir: Path | None = None,
    snapshot_loader: Callable[[Scenario], IncidentSnapshot] | None = None,
    out_dir: Path | None = None,
    recall_gate: float = 1.0,
    log: Callable[[str], None] = print,
) -> HeadToHeadResult:
    """Run every arm on every scenario; refuse to start without parity."""
    specs, probe = build_arm_specs(llm_factory, arms)
    fingerprint = assert_parity(*specs)  # raises ParityViolation on any mismatch
    log(f"parity OK — environment fingerprint {fingerprint[:16]}…")

    result = HeadToHeadResult(
        environment_fingerprint=fingerprint,
        model=probe.model,
        arms={spec.name: spec.describe() for spec in specs},
    )

    records_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        records_path = out_dir / "records.jsonl"
        records_path.write_text("")  # truncate previous run

    loader = snapshot_loader or (lambda s: load_snapshot(s, cache_dir))

    for scenario in scenarios:
        snapshot = loader(scenario)
        for arm in arms:
            llm = llm_factory()  # fresh client per run: no cross-arm state
            toolbox = SnapshotToolbox(snapshot)
            started = time.time()
            run = arm.run(llm, toolbox)
            elapsed = time.time() - started

            score = score_task(
                scenario.ground_truth,
                [Prediction(name=p["name"], kind=p.get("kind", "")) for p in run.predictions],
                escalated=run.escalated,
                num_turns=run.num_turns,
                num_tool_calls=run.num_tool_calls,
                stop_reason=run.stop_reason,
                recall_gate=recall_gate,
            )
            result.task_scores.setdefault(arm.name, []).append(score)

            record = {
                "scenario_id": scenario.scenario_id,
                "arm": arm.name,
                "elapsed_s": round(elapsed, 2),
                "run": run.to_dict(),
                "score": score.to_dict(),
            }
            if records_path is not None:
                with records_path.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
            log(
                f"{scenario.scenario_id} [{arm.name}] score={score.score:.2f} "
                f"P={score.precision:.2f} R={score.recall:.2f} turns={run.num_turns} "
                f"stop={run.stop_reason}"
            )

    if out_dir is not None:
        (out_dir / "summary.json").write_text(json.dumps(result.summary(), indent=2))
    return result
