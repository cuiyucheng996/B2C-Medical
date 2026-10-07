"""用户沟通事件：图节点输出 → API 渲染 SSE / 多通道推送（纯类型）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

CommKind = Literal["reply", "card", "notification"]

CardType = Literal[
    "body_select",
    "disease_select",
    "symptom_select",
    "consent",
]

NotificationLevel = Literal["info", "warning", "success"]

__all__ = [
    "CardType",
    "CommEvent",
    "CommKind",
    "NotificationLevel",
]


class CommEvent(BaseModel):
    """一条对用户可见的沟通意图（与 AgentEvent 输入信封对称）。"""

    comm_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: CommKind
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]
