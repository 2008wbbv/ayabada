"""Scripted LLM for offline tests and dry runs.

A :class:`ScriptedLLM` replays a fixed sequence of responses regardless of
input, letting the loop, policy and runner be exercised end-to-end with no
API key and full determinism. Script entries are written in a compact form
and expanded into the same :class:`LLMResponse` shape the real client emits.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from ayabada.llm.base import LLMResponse, ToolCall


@dataclass
class ScriptedLLM:
    """Pops one scripted response per ``complete()`` call.

    Each script entry is either a plain string (a text-only response) or a
    dict: ``{"text": ..., "tool": name, "input": {...}}``. When the script is
    exhausted the last entry repeats, so tests can't hang the loop.
    """

    script: list[Any]
    model: str = "scripted-model"
    model_params: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        index = min(len(self.calls) - 1, len(self.script) - 1)
        entry = self.script[index]
        if isinstance(entry, str):
            return LLMResponse(
                text=entry,
                content_blocks=[{"type": "text", "text": entry}],
            )
        text = entry.get("text", "")
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict[str, Any]] = []
        if text:
            content_blocks.append({"type": "text", "text": text})
        if "tool" in entry:
            call_id = f"toolu_mock_{next(self._counter)}"
            tool_input = entry.get("input", {})
            tool_calls.append(ToolCall(id=call_id, name=entry["tool"], input=tool_input))
            content_blocks.append(
                {"type": "tool_use", "id": call_id, "name": entry["tool"], "input": tool_input}
            )
        return LLMResponse(
            text=text,
            tool_calls=tuple(tool_calls),
            stop_reason="tool_use" if tool_calls else "end_turn",
            content_blocks=content_blocks,
        )
