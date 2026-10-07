"""Agent 事件信封：统一描述进入问诊系统的各类输入（纯类型，无 LangGraph）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

EventType = Literal[
    "user.input",
    "user.interrupt",
    "tool.result",
    "system.event",
    "timer.trigger",
]

EventChannel = Literal[
    "web_step",
    "web_stream",
    "webhook",
    "cron",
    "internal",
]

EventPriority = Literal["urgent", "normal", "low"]

UserInputKind = Literal[
    "text",
    "body_option",
    "disease_option",
    "symptom_option",
    "consent",
]

__all__ = [
    "AgentEvent",
    "EventChannel",
    "EventPriority",
    "EventType",
    "UserInputKind",
]


class AgentEvent(BaseModel):
    """进入问诊系统的一条结构化事件。"""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    channel: EventChannel = "web_step"
    priority: EventPriority = "normal"
    thread_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]
