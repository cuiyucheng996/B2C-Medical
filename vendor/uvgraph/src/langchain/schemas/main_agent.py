"""主 Agent 结构化输出。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...schemas import Artifact, BodyPart, MainAgentPlan, SubAgentConclusion, SubAgentResult
from ...schemas.tool_trace import ToolTraceEntry

__all__ = [
    "MainAgentComposeOutput",
    "MainAgentOutput",
    "MainAgentRunResult",
]


class MainAgentOutput(BaseModel):
    """主 Agent 结构化结论：听懂意图 + 路由字段 + 人话答复要点。"""

    intent: str = Field(
        description="用户本轮意图摘要，如「描述头痛并询问可能疾病」「确认喘不上气症状」",
    )
    is_relevant: bool = Field(description="是否与当前问诊阶段任务相关")
    is_chitchat: bool = Field(default=False, description="是否纯寒暄/无关话题")
    is_sensitive: bool = Field(default=False, description="是否敏感违规内容")
    needs_clarification: bool = Field(default=False, description="是否描述过短模糊需追问")
    reason: str = Field(default="", description="简短判定理由")
    # 第一大轮
    body_part: BodyPart | None = Field(
        default=None,
        description="第一大轮：无已选部位时推断部位；已有部位时保持 null",
    )
    extracted_symptoms: list[str] = Field(default_factory=list, description="提取的症状关键词")
    disease_suggested: list[str] = Field(
        default_factory=list,
        description="第一大轮：Tool 检索或推断后的候选疾病名",
    )
    # 第二大轮
    matched: bool = Field(default=False, description="第二大轮：是否高置信命中官方症状")
    confirmed_symptom: str | None = Field(default=None, description="第二大轮：锁定的官方症状名")
    candidate_symptoms: list[str] = Field(
        default_factory=list,
        description="第二大轮：未精确命中时的 top 候选（最多 3）",
    )
    normalized_description: str = Field(default="", description="第二大轮：标准化后的症状描述")
    # 组织答复
    assistant_reply: str = Field(
        description="给用户的人话结论或下一步引导，专业亲和，不捏造未确认信息",
    )


class MainAgentComposeOutput(BaseModel):
    """主 Agent 答复环节：根据计划 + 子 Agent 结论组织人话。"""

    assistant_reply: str = Field(
        description="面向用户的专业、亲和结论或下一步引导，不暴露内部字段名",
    )


class MainAgentRunResult(BaseModel):
    """主 Agent 一次 thinking 运行的完整结果（含 Tool 轨迹）。"""

    major_round: int
    output: MainAgentOutput
    plan: MainAgentPlan | None = None
    sub_result: SubAgentResult | None = None
    sub_conclusion: SubAgentConclusion | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    sub_agent_refs: list[str] = Field(default_factory=list)
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    rag_context: list[str] = Field(default_factory=list)
    used_llm: bool = True
