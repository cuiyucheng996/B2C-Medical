"""流式 tool_calls 累积器：按 index 拼接 arguments 碎片，拼完整后再 json.loads。

状态机：等待首片（记 id/name）→ 累积 arguments 碎片 → 流结束 flush 产出 ToolCall。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CompletedToolCall",
    "ToolCallsAccumulator",
]


@dataclass(frozen=True)
class CompletedToolCall:
    """arguments 已拼完整且通过 json.loads 校验的 tool call。"""

    index: int
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict[str, Any]:
        if not self.arguments.strip():
            return {}
        parsed = json.loads(self.arguments)
        if not isinstance(parsed, dict):
            raise ValueError(f"tool arguments must be object, got {type(parsed)}")
        return parsed


@dataclass
class _ToolCallSlot:
    index: int
    id: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)

    def arguments_text(self) -> str:
        return "".join(self.argument_parts)


class ToolCallsAccumulator:
    """按 ``index`` 分组累积流式 tool_calls；仅在 JSON 完整可解析时产出。"""

    def __init__(self) -> None:
        self._slots: dict[int, _ToolCallSlot] = {}

    def reset(self) -> None:
        self._slots.clear()

    def feed_openai_delta(self, tool_calls: list[dict[str, Any]]) -> list[CompletedToolCall]:
        """消费 OpenAI 兼容 ``delta.tool_calls`` 分片（单 chunk 内列表）。"""
        fragments: list[dict[str, Any]] = []
        for item in tool_calls or []:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", 0))
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            fragments.append(
                {
                    "index": index,
                    "id": item.get("id"),
                    "name": fn.get("name"),
                    "args": fn.get("arguments") or "",
                }
            )
        return self._feed_fragments(fragments, finalize=False)

    def feed_delta(self, delta: Any) -> list[CompletedToolCall]:
        """消费统一 ToolCallDelta（index / id / function_name / function_arguments）。"""
        return self._feed_fragments(
            [
                {
                    "index": int(getattr(delta, "index", 0) or 0),
                    "id": getattr(delta, "id", None),
                    "name": getattr(delta, "function_name", None),
                    "args": getattr(delta, "function_arguments", None) or "",
                }
            ],
            finalize=False,
        )

    def feed_langchain_chunk(self, chunk: Any) -> list[CompletedToolCall]:
        """消费 LangChain ``AIMessageChunk.tool_call_chunks``。"""
        fragments: list[dict[str, Any]] = []
        for item in getattr(chunk, "tool_call_chunks", None) or []:
            if isinstance(item, dict):
                fragments.append(
                    {
                        "index": int(item.get("index", 0)),
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "args": item.get("args") or "",
                    }
                )
            else:
                fragments.append(
                    {
                        "index": int(getattr(item, "index", 0)),
                        "id": getattr(item, "id", None),
                        "name": getattr(item, "name", None),
                        "args": getattr(item, "args", "") or "",
                    }
                )
        return self._feed_fragments(fragments, finalize=False)

    def flush(self) -> list[CompletedToolCall]:
        """当前轮流结束：尝试把已累积的 slots 解析为完整 ToolCall。"""
        completed: list[CompletedToolCall] = []
        for index in sorted(self._slots):
            call = self._finalize_slot(index)
            if call is not None:
                completed.append(call)
        self._slots.clear()
        return completed

    def _feed_fragments(
        self,
        fragments: list[dict[str, Any]],
        *,
        finalize: bool,
    ) -> list[CompletedToolCall]:
        emitted: list[CompletedToolCall] = []
        for frag in fragments:
            index = int(frag.get("index", 0))
            slot = self._slots.setdefault(index, _ToolCallSlot(index=index))
            if frag.get("id"):
                slot.id = str(frag["id"])
            if frag.get("name"):
                slot.name = str(frag["name"])
            args_piece = frag.get("args")
            if args_piece:
                slot.argument_parts.append(str(args_piece))
            if finalize:
                call = self._finalize_slot(index)
                if call is not None:
                    emitted.append(call)
        return emitted

    def _finalize_slot(self, index: int) -> CompletedToolCall | None:
        slot = self._slots.get(index)
        if slot is None:
            return None
        if not slot.id or not slot.name:
            logger.debug("tool call slot incomplete (missing id/name): index=%s", index)
            return None
        args_text = slot.arguments_text()
        if not args_text.strip():
            logger.debug("tool call slot incomplete (empty arguments): index=%s name=%s", index, slot.name)
            return None
        try:
            json.loads(args_text)
        except json.JSONDecodeError:
            logger.debug(
                "tool call arguments not valid json yet: index=%s name=%s text=%r",
                index,
                slot.name,
                args_text,
            )
            return None
        return CompletedToolCall(
            index=index,
            id=slot.id,
            name=slot.name,
            arguments=args_text,
        )


if __name__ == "__main__":
    acc = ToolCallsAccumulator()
    chunks = [
        [{"index": 0, "id": "call_001", "function": {"name": "get_grid_info", "arguments": ""}}],
        [{"index": 0, "function": {"arguments": '{"name":'}}],
        [{"index": 0, "function": {"arguments": '"上海"}'}}],
    ]
    for part in chunks:
        acc.feed_openai_delta(part)
    done = acc.flush()
    assert len(done) == 1
    assert done[0].name == "get_grid_info"
    assert done[0].parsed_arguments() == {"name": "上海"}
    print("ok:", done[0])
