"""事件总线：emit 给 SSE 订阅方，yield 给直接消费循环的调用方。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["AgentEvent", "EventBus", "EventType"]


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TEXT_CHUNK = "text_chunk"
    THINK_CHUNK = "think_chunk"
    TOOL_CALLS_DELTA = "tool_calls_delta"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ARTIFACT = "artifact"
    SUSPENDED = "suspended"
    DONE = "done"
    ERROR = "error"


@dataclass(slots=True)
class AgentEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> dict[str, Any]:
        """兼容现有问诊 SSE 字段，同时保留细分事件名。"""
        payload: dict[str, Any] = {"event": self.type.value, **self.data}
        if self.type is EventType.TEXT_CHUNK:
            payload["event"] = "token"
            payload.setdefault("content", self.data.get("content", ""))
        elif self.type is EventType.THINK_CHUNK:
            payload["event"] = "think_chunk"
            payload.setdefault("content", self.data.get("content", ""))
            payload.setdefault("message", self.data.get("content", ""))
        elif self.type is EventType.TOOL_CALLS_DELTA:
            payload["event"] = "tool_call_fragment"
            payload.setdefault("arguments_fragment", self.data.get("function_arguments") or "")
        elif self.type is EventType.TOOL_START:
            payload["event"] = "tool_call"
            payload.setdefault("status", "ready")
        elif self.type is EventType.TOOL_END:
            payload["event"] = "tool_end"
        elif self.type is EventType.ARTIFACT:
            payload["event"] = "artifact"
        elif self.type is EventType.AGENT_START:
            payload["event"] = "thinking"
            payload.setdefault("message", "正在分析…")
        elif self.type is EventType.DONE:
            payload["event"] = "agent_done"
        return payload


EventListener = Callable[[AgentEvent], None]


class EventBus:
    """进程内事件总线（B/C 端都可订阅）。"""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def create_emit_fn(self, source: str) -> Callable[[dict[str, Any]], None]:
        """子 Agent 用的 emit：打上来源，中间 token 进思考区。"""

        def _emit(payload: dict[str, Any]) -> None:
            data = {k: v for k, v in payload.items() if k != "event"}
            data["source"] = source
            raw = str(payload.get("event") or "token")
            if source.startswith("subagent:") and raw in {"token", "text_chunk"}:
                event_type = EventType.THINK_CHUNK
                data.setdefault("content", payload.get("content") or payload.get("message") or "")
            elif raw in {"token", "text_chunk"}:
                event_type = EventType.TEXT_CHUNK
            elif raw in {"think_chunk", "thinking"}:
                event_type = EventType.THINK_CHUNK
                data.setdefault("content", payload.get("content") or payload.get("message") or "")
            elif raw in {"tool_call", "tool_start"}:
                event_type = EventType.TOOL_START
            elif raw == "tool_end":
                event_type = EventType.TOOL_END
            elif raw == "artifact":
                event_type = EventType.ARTIFACT
            else:
                event_type = EventType.THINK_CHUNK
                data.setdefault("content", str(payload.get("content") or payload.get("message") or ""))
            self.emit(AgentEvent(type=event_type, data=data))

        return _emit
