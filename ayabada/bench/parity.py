"""Parity guardrails: the one rule that can't break.

In any scored run the base model and the tool access must be identical
between the confidence arm and the baseline. Vary only the architecture.
If one arm has any tool, context or model advantage, a win proves nothing.

This module makes that rule executable instead of aspirational:

- Every arm declares an :class:`ArmSpec` — the environment it runs in (model
  id, sampling/thinking config, the exact tool schemas it is given, the task
  instructions) plus the architecture knobs (which *are* allowed to differ).
- ``environment_fingerprint()`` hashes only the must-match parts.
- :func:`assert_parity` refuses to proceed when fingerprints differ, and the
  runner stores the shared fingerprint in every results file so a published
  number can be audited back to "same model, same tools".

Things the product is allowed to do but a scored run is not (web browsing
for context, model switching/handoff, extra tools) all show up as either a
tool-schema difference or a model difference, so they are caught here by
construction rather than by code review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class ParityViolation(RuntimeError):
    """Raised when two arms of a scored run are not environment-identical."""


@dataclass(frozen=True)
class ArmSpec:
    """Everything that defines one arm of a head-to-head run."""

    name: str
    model: str
    tool_schemas: tuple[dict[str, Any], ...]
    task_instructions: str
    model_params: dict[str, Any] = field(default_factory=dict)
    # Architecture is the experimental variable — excluded from the
    # environment fingerprint but logged alongside it.
    architecture: str = "baseline"
    architecture_params: dict[str, Any] = field(default_factory=dict)

    def environment_fingerprint(self) -> str:
        """Hash of the must-match surface: model + params + tools + task."""
        payload = {
            "model": self.model,
            "model_params": _canonical(self.model_params),
            "tools": _canonical(sorted(self.tool_schemas, key=lambda t: t.get("name", ""))),
            "task_instructions": self.task_instructions,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "model_params": self.model_params,
            "tools": sorted(t.get("name", "?") for t in self.tool_schemas),
            "architecture": self.architecture,
            "architecture_params": self.architecture_params,
            "environment_fingerprint": self.environment_fingerprint(),
        }


def _canonical(obj: Any) -> Any:
    """Round-trip through JSON to normalize types for stable hashing."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def assert_parity(*arms: ArmSpec) -> str:
    """Verify every arm shares one environment fingerprint; return it.

    Called by the runner before any scored task executes. Failing loudly here
    is the design: a run that would produce an unpublishable comparison never
    starts.
    """
    if not arms:
        raise ParityViolation("no arms supplied")
    fingerprints = {arm.name: arm.environment_fingerprint() for arm in arms}
    unique = set(fingerprints.values())
    if len(unique) > 1:
        detail = "\n".join(
            f"  {arm.name}: model={arm.model} tools={sorted(t.get('name', '?') for t in arm.tool_schemas)} fp={fp}"
            for arm, fp in ((a, fingerprints[a.name]) for a in arms)
        )
        raise ParityViolation(
            "Scored-run parity violation — arms differ in model, model params, "
            "tools, or task instructions (only architecture may vary):\n" + detail
        )
    return unique.pop()
