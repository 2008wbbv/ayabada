"""Anthropic-backed LLM client.

The base model stays stock: one ``messages.create`` per turn, adaptive
thinking, no sampling parameters, no server-side tools. Everything the model
can see or do is decided by the agent loop that owns the conversation, which
is what keeps the two arms of a scored run comparable.
"""

from __future__ import annotations

from typing import Any

from ayabada.llm.base import LLMResponse, ToolCall

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16000,
        thinking: dict[str, Any] | None = None,
    ) -> None:
        import anthropic  # deferred so the package works without the extra installed

        self._client = anthropic.Anthropic()
        self.model = model
        self._max_tokens = max_tokens
        self._thinking = thinking if thinking is not None else {"type": "adaptive"}
        self.model_params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "thinking": self._thinking,
        }

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            thinking=self._thinking,
            system=system,
            messages=messages,
            tools=tools,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content_blocks: list[dict[str, Any]] = []
        for block in response.content:
            content_blocks.append(block.model_dump())
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tuple(tool_calls),
            stop_reason=response.stop_reason or "end_turn",
            content_blocks=content_blocks,
        )
