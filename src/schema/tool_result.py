"""陪聊工具回给模型的统一信封。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["CompanionToolEnvelope"]


class CompanionToolEnvelope(BaseModel):
    ok: bool = Field(description="工具是否正常执行完")
    tool: str = Field(description="工具名")
    usable: bool = Field(description="是否有可组句的证据")
    retryable: bool = Field(description="效果不好时模型可否换参数再调")
    error: str = Field(default="", description="失败原因")
    summary: str = Field(default="", description="一句话摘要，给模型甄别用")
    data: dict[str, Any] = Field(default_factory=dict, description="原始证据，按工具而异")
