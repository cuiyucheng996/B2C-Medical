"""症状匹配 Agent 结构化输入/输出。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas import Artifact
from ...schemas.tool_trace import ToolTraceEntry

__all__ = [
    "SymptomAnalysis",
    "SymptomCheckResult",
]


class SymptomAnalysis(BaseModel):
    """Agent 结构化输出：口语描述 → 疾病内官方症状。"""

    matched: bool = Field(description="是否已在已锁定疾病症状列表中精确或高置信命中")
    confirmed_symptom: str | None = Field(
        default=None,
        description="若 matched=true，写入最符合的官方症状名；否则 null",
    )
    candidate_symptoms: list[str] = Field(
        default_factory=list,
        description="未精确命中时，给出最多 3 个最接近的官方症状候选",
    )
    normalized_description: str = Field(
        default="",
        description="去掉「没有太像的」等口语前缀后的症状描述",
    )
    needs_clarification: bool = Field(
        default=False,
        description="描述过短或过于模糊，需追问",
    )
    reason: str = Field(default="", description="简短判定理由")


class SymptomCheckResult(BaseModel):
    """节点层使用的症状匹配结果（含规则兜底与 tool 轨迹）。"""

    matched: bool = False
    confirmed_symptom: str | None = None
    normalized_symptoms: list[str] = Field(default_factory=list)
    candidate_top3: list[str] = Field(default_factory=list)
    rag_context: list[str] = Field(default_factory=list)
    reason: str = ""
    needs_clarification: bool = False
    is_relevant: bool = True
    used_llm: bool = False
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    sub_agent_refs: list[str] = Field(default_factory=list)
