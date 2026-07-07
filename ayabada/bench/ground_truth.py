"""Parse ITBench-AA ground-truth YAML into a scoreable structure.

Two shapes exist in the released data: early scenarios put ``fault`` /
``groups`` / ``aliases`` at the top level, later ones wrap them in an
``apiVersion/kind/metadata/spec`` envelope. Both parse to the same
:class:`GroundTruth`.

Scoring semantics (see :mod:`ayabada.bench.scoring`):

- The *positive set* is the set of root-cause entity groups. A group is
  root-cause if it carries ``root_cause: true``; when no group is marked,
  the groups matching the ``fault`` entities (by kind + name-regex) are used.
- ``aliases`` merge groups that refer to the same underlying entity (e.g. a
  service and its pod), so predicting either alias counts once, not twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class EntityGroup:
    id: str
    kind: str = ""
    namespace: str = ""
    filters: tuple[str, ...] = ()
    root_cause: bool = False

    def matches(self, name: str, kind: str | None = None) -> bool:
        """True when a raw entity name (and optional kind) maps to this group."""
        if kind and self.kind and kind.lower() != self.kind.lower():
            return False
        for pattern in self.filters:
            try:
                if re.fullmatch(pattern, name) or re.match(pattern, name):
                    return True
            except re.error:
                if pattern == name:
                    return True
        return False


@dataclass(frozen=True)
class FaultEntity:
    name: str
    kind: str = ""
    category: str = ""
    condition: str = ""
    fault_mechanism: str = ""


@dataclass
class GroundTruth:
    scenario_id: str
    fault_entities: list[FaultEntity] = field(default_factory=list)
    groups: list[EntityGroup] = field(default_factory=list)
    # Each alias set is a group of group-ids naming the same underlying entity.
    aliases: list[frozenset[str]] = field(default_factory=list)
    expected_alerts: list[dict[str, Any]] = field(default_factory=list)
    propagations: list[dict[str, Any]] = field(default_factory=list)
    recommended_actions: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Canonicalization
    # ------------------------------------------------------------------
    def canonical_id(self, group_id: str) -> str:
        """Collapse a group id to a stable representative of its alias set."""
        for alias_set in self.aliases:
            if group_id in alias_set:
                return min(alias_set)
        return group_id

    def resolve(self, name: str, kind: str | None = None) -> str | None:
        """Map a raw entity name (e.g. ``adservice-554b-lwpht``) to a canonical
        group id, or None if it matches no known group."""
        candidates = [g for g in self.groups if g.matches(name, kind)]
        if not candidates:
            # Also accept a literal group id.
            if any(g.id == name for g in self.groups):
                return self.canonical_id(name)
            return None
        # Prefer the most specific filter (longest pattern) on ties.
        best = max(candidates, key=lambda g: max((len(p) for p in g.filters), default=0))
        return self.canonical_id(best.id)

    def root_cause_ids(self) -> set[str]:
        """Canonical ids of the true root-cause entities (the positive set)."""
        marked = {self.canonical_id(g.id) for g in self.groups if g.root_cause}
        if marked:
            return marked
        # Fall back to mapping the fault entities onto groups.
        ids: set[str] = set()
        for fault in self.fault_entities:
            resolved = self.resolve(fault.name, fault.kind or None)
            ids.add(resolved if resolved else fault.name)
        return ids


def parse_ground_truth(text: str, scenario_id: str = "") -> GroundTruth:
    doc = yaml.safe_load(text) or {}
    if "spec" in doc and isinstance(doc["spec"], dict):
        body = doc["spec"]
        scenario_id = scenario_id or str(doc.get("metadata", {}).get("name", ""))
    else:
        body = doc

    fault_entities: list[FaultEntity] = []
    for item in body.get("fault") or []:
        entity = item.get("entity") or {}
        name = str(entity.get("name", "")).strip()
        if not name:
            # "changed" faults (e.g. a ConfigMap edit) put identity one level in.
            changed = entity.get("changed") or {}
            name = str(changed.get("name", "")).strip()
            entity = {**entity, "kind": changed.get("kind", entity.get("kind", ""))}
        if not name:
            continue
        fault_entities.append(
            FaultEntity(
                name=name,
                kind=str(entity.get("kind", "")),
                category=str(item.get("category", "")),
                condition=str(item.get("condition", "")),
                fault_mechanism=str(item.get("fault_mechanism", "")),
            )
        )

    groups = [
        EntityGroup(
            id=str(g.get("id", "")),
            kind=str(g.get("kind", "")),
            namespace=str(g.get("namespace", "")),
            filters=tuple(str(f) for f in (g.get("filter") or [])),
            root_cause=bool(g.get("root_cause", False)),
        )
        for g in body.get("groups") or []
        if g.get("id")
    ]

    aliases = [
        frozenset(str(a) for a in alias_set)
        for alias_set in body.get("aliases") or []
        if alias_set
    ]

    return GroundTruth(
        scenario_id=scenario_id,
        fault_entities=fault_entities,
        groups=groups,
        aliases=aliases,
        expected_alerts=list(body.get("alerts") or []),
        propagations=list(body.get("propagations") or []),
        recommended_actions=list(body.get("recommendedActions") or []),
    )
