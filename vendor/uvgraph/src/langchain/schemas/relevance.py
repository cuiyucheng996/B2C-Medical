"""相关性 Agent 结构化输入/输出。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas import Artifact, BodyPart
from ...schemas.tool_trace import ToolTraceEntry

__all__ = [
    "RelevanceAnalysis",
    "RelevanceResult",
]


class RelevanceAnalysis(BaseModel):
    """Agent 结构化输出（LLM 分析结果）。"""

    is_relevant: bool = Field(description="是否与当前身体不适/症状相关")
    is_sensitive: bool = Field(default=False, description="是否含敏感违规内容")
    is_chitchat: bool = Field(default=False, description="是否仅为寒暄闲聊、与病情无关")
    body_part: BodyPart | None = Field(
        default=None,
        description="若上下文无已选部位且可判断不适部位，从八大部位中选一项；否则 null",
    )
    reason: str = Field(default="", description="简短判定理由，供路由与日志使用")
    extracted_symptoms: list[str] = Field(
        default_factory=list,
        description="从用户话术中提取的症状关键词",
    )


class RelevanceResult(BaseModel):
    """节点层使用的判定结果（含规则兜底与 Tool RAG 字段）。"""

    is_relevant: bool
    is_sensitive: bool = False
    is_chitchat: bool = False
    body_part: BodyPart | None = None
    reason: str = ""
    extracted_symptoms: list[str] = Field(default_factory=list)
    used_llm: bool = False
    disease_suggested: list[str] = Field(default_factory=list)
    rag_context: list[str] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    sub_agent_refs: list[str] = Field(default_factory=list)
