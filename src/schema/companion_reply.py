"""陪聊组句结构化输出。"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["CompanionReply"]


class CompanionReply(BaseModel):
    assistant_reply: str = Field(description="给用户看的中文回复，不要 JSON")
