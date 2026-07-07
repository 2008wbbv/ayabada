"""Investigation tools over an IncidentSnapshot.

This is the *entire* action surface an agent gets during a scored run, for
both arms — the toolbox is constructed once per task and handed to each arm
unchanged, so the parity fingerprint covers exactly these schemas. Tools are
read-only views of the frozen snapshot; nothing here can reach the network,
the live cluster, or the web.

``submit_diagnosis`` is the terminal tool: calling it ends the run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from ayabada.snapshot import IncidentSnapshot

MAX_RESULT_CHARS = 8000


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n... [truncated, {len(text)} chars total]"


class SubmitDiagnosis(Exception):
    """Control-flow signal: the agent called the terminal tool."""

    def __init__(self, entities: list[dict[str, str]], summary: str) -> None:
        super().__init__("diagnosis submitted")
        self.entities = entities
        self.summary = summary


class SnapshotToolbox:
    """Executable tools + their JSON schemas for one snapshot."""

    def __init__(self, snapshot: IncidentSnapshot) -> None:
        self.snapshot = snapshot
        self._handlers: dict[str, Callable[..., str]] = {
            "get_alerts": self._get_alerts,
            "list_entities": self._list_entities,
            "get_events": self._get_events,
            "list_telemetry_files": self._list_telemetry_files,
            "read_telemetry": self._read_telemetry,
            "grep_telemetry": self._grep_telemetry,
            "submit_diagnosis": self._submit_diagnosis,
        }

    # ------------------------------------------------------------------
    # Schemas (Anthropic tool format)
    # ------------------------------------------------------------------
    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": "get_alerts",
                "description": (
                    "List the firing alerts in the incident snapshot: alert name, "
                    "severity, labels (service/namespace), description and value."
                ),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "list_entities",
                "description": (
                    "List entity names (pods, services, ...) visible in the snapshot telemetry."
                ),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "get_events",
                "description": (
                    "Kubernetes events from the snapshot, most recent first. Optionally "
                    "filter by a substring of the involved object's name and by event "
                    "type ('Warning' or 'Normal')."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Substring filter on object name"},
                        "event_type": {"type": "string", "enum": ["Warning", "Normal"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_telemetry_files",
                "description": "List raw telemetry files available in the snapshot (metrics, logs, traces).",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "read_telemetry",
                "description": "Read lines from a raw telemetry file in the snapshot.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path from list_telemetry_files"},
                        "start_line": {"type": "integer", "minimum": 0},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 500},
                    },
                    "required": ["file"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "grep_telemetry",
                "description": "Search a raw telemetry file for a regex; returns matching lines.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "file": {"type": "string", "description": "Path from list_telemetry_files"},
                        "max_matches": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "required": ["pattern", "file"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "submit_diagnosis",
                "description": (
                    "Submit your final root-cause diagnosis and end the investigation. "
                    "Name ONLY the entities you believe caused the incident — extra "
                    "entities beyond the true root cause reduce the score."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "kind": {
                                        "type": "string",
                                        "description": "Kubernetes kind, e.g. Pod, Service, Deployment, ConfigMap",
                                    },
                                },
                                "required": ["name"],
                                "additionalProperties": False,
                            },
                        },
                        "summary": {"type": "string", "description": "One-paragraph root-cause explanation"},
                    },
                    "required": ["entities", "summary"],
                    "additionalProperties": False,
                },
            },
        ]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def execute(self, name: str, tool_input: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            return _truncate(handler(**tool_input))
        except SubmitDiagnosis:
            raise
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # tool errors go back to the model, not up the stack
            return f"Error: {type(exc).__name__}: {exc}"

    def _get_alerts(self) -> str:
        if not self.snapshot.alerts:
            return "No alerts in snapshot."
        return json.dumps([a.to_dict() for a in self.snapshot.alerts], indent=1)

    def _list_entities(self) -> str:
        names = self.snapshot.entity_names()
        return "\n".join(names) if names else "No entities identified in snapshot."

    def _get_events(self, entity: str = "", event_type: str = "", limit: int = 50) -> str:
        rows = list(self.snapshot.events)
        if entity:
            rows = [e for e in rows if entity.lower() in str(e.get("object", "")).lower()]
        if event_type:
            rows = [e for e in rows if e.get("type", "") == event_type]
        rows = rows[-limit:]
        if not rows:
            return "No matching events."
        lines = [
            f"{e.get('timestamp','')} [{e.get('type','')}/{e.get('reason','')}] "
            f"{e.get('kind','')}/{e.get('object','')} x{e.get('count',1)}: {e.get('message','')}"
            for e in rows
        ]
        return "\n".join(lines)

    def _telemetry_root(self) -> Path | None:
        return self.snapshot.raw_dir

    def _resolve_file(self, file: str) -> Path:
        root = self._telemetry_root()
        if root is None:
            raise FileNotFoundError("snapshot has no raw telemetry files")
        target = (root / file).resolve()
        if not target.is_relative_to(root.resolve()):
            raise PermissionError("path escapes the snapshot directory")
        if not target.is_file():
            raise FileNotFoundError(f"no such telemetry file: {file}")
        return target

    def _list_telemetry_files(self) -> str:
        root = self._telemetry_root()
        if root is None or not root.exists():
            return "No raw telemetry files in this snapshot."
        lines = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root)
                lines.append(f"{rel} ({path.stat().st_size} bytes)")
        return "\n".join(lines) if lines else "No raw telemetry files in this snapshot."

    def _read_telemetry(self, file: str, start_line: int = 0, max_lines: int = 50) -> str:
        target = self._resolve_file(file)
        out: list[str] = []
        with target.open(errors="replace") as fh:
            for i, line in enumerate(fh):
                if i < start_line:
                    continue
                if len(out) >= max_lines:
                    break
                out.append(f"{i}: {line.rstrip()}")
        return "\n".join(out) if out else f"(no lines at offset {start_line})"

    def _grep_telemetry(self, pattern: str, file: str, max_matches: int = 50) -> str:
        target = self._resolve_file(file)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        out: list[str] = []
        with target.open(errors="replace") as fh:
            for i, line in enumerate(fh):
                if rx.search(line):
                    out.append(f"{i}: {line.rstrip()}")
                    if len(out) >= max_matches:
                        break
        return "\n".join(out) if out else "(no matches)"

    def _submit_diagnosis(self, entities: list[dict[str, str]], summary: str) -> str:
        cleaned = [
            {"name": str(e.get("name", "")).strip(), "kind": str(e.get("kind", "")).strip()}
            for e in entities
            if str(e.get("name", "")).strip()
        ]
        raise SubmitDiagnosis(cleaned, summary)
