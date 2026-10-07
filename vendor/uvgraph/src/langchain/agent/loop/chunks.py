"""LLM 流式分片：text / think / tool_calls_delta / done。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.llm.tool_calls_accumulator import CompletedToolCall

__all__ = ["LLMChunk", "ToolCallDelta"]

ChunkKind = Literal["text", "think", "tool_calls_delta", "done"]


@dataclass(slots=True)
class ToolCallDelta:
    index: int = 0
    id: str | None = None
    function_name: str | None = None
    function_arguments: str | None = None


@dataclass(slots=True)
class LLMChunk:
    kind: ChunkKind
    text: str = ""
    tool_calls_delta: ToolCallDelta | None = None
    full_text: str = ""
    full_think: str = ""
    tool_calls: list[CompletedToolCall] | None = None
    extra: dict[str, Any] | None = None
