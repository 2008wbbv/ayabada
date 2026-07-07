"""Minimal LLM client abstraction.

Messages use the Anthropic Messages API shape (role + content blocks) as the
lingua franca, so the real client is a thin passthrough and the scripted mock
consumes exactly what the real model would. Both arms of a scored run receive
an :class:`LLMClient` constructed from the *same* spec — the client carries
no arm-specific state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"
    # The assistant content blocks to append verbatim to history (preserves
    # thinking blocks / tool_use ids for the real API; synthesized for mocks).
    content_blocks: list[dict[str, Any]] = field(default_factory=list)


class LLMClient(Protocol):
    """One completion call; the agent loop owns the conversation."""

    model: str
    model_params: dict[str, Any]

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...
