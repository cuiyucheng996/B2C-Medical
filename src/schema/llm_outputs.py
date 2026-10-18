"""第三方 LLM 纯 JSON / structured 输出模型（与 CompanionReply 工具形输出分开）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "CompanionReviewOutput",
    "CompanionThinkOutput",
    "FactExtractOutput",
    "FactItem",
    "GraphitiEntityItem",
    "GraphitiNeo4jExtractOutput",
    "IntentMajorOutput",
    "IntentFinePick",
    "IntentFineMultiOutput",
    "IntentFineScoreItem",
    "KnowledgeQueryPlan",
]

ThinkAction = Literal["compose", "retry_tools", "give_up"]
FactKind = Literal[
    "place",
    "time",
    "mood",
    "weather",
    "event",
    "occupation",
    "habit",
    "person",
    "thing",
    "negation",
    "other",
]


class CompanionThinkOutput(BaseModel):
    fit: bool = Field(description="usable 的 data 能否直接回答用户问题")
    action: ThinkAction = Field(description="compose=组句；retry_tools=换参再调；give_up=停止工具如实说")
    reason: str = Field(default="", max_length=80, description="简短理由")


class CompanionReviewOutput(BaseModel):
    pass_: bool = Field(alias="pass", description="草稿是否通过评审")
    reason: str = Field(default="", max_length=80)
    hint: str = Field(default="", max_length=120, description="不通过时给执行 Agent 的改写要点")

    model_config = {"populate_by_name": True}


class FactItem(BaseModel):
    name: str = Field(description="短名")
    fact: str = Field(description="一句陈述")
    kind: FactKind = Field(default="other")
    negated: bool = Field(default=False)


class FactExtractOutput(BaseModel):
    facts: list[FactItem] = Field(default_factory=list)


class GraphitiEntityItem(BaseModel):
    entity_type: str = Field(description="Graphiti 类型或 部位/疾病/症状")
    name: str = Field(description="标准短名")
    mention: str = Field(default="", description="原文片段")
    negated: bool = Field(default=False)


class GraphitiNeo4jExtractOutput(BaseModel):
    entities: list[GraphitiEntityItem] = Field(default_factory=list)


class IntentMajorOutput(BaseModel):
    major: str = Field(description="大类名称")
    reason: str = Field(default="", description="短理由")


class IntentFinePick(BaseModel):
    intent: str = Field(description="清单中的细意图名或未细分")
    reason: str = Field(default="", description="短理由")


class IntentFineScoreItem(BaseModel):
    intent: str = Field(description="清单中的细意图名")
    confidence: float = Field(ge=0.0, le=1.0, description="0~1 相关度")
    reason: str = Field(default="", description="短理由")


class IntentFineMultiOutput(BaseModel):
    items: list[IntentFineScoreItem] = Field(default_factory=list)


class KnowledgeQueryPlan(BaseModel):
    """公共知识问答进子 Agent 前的检索计划。"""

    split: bool = Field(description="是否拆成多个独立 query")
    queries: list[str] = Field(default_factory=list, description="重述后的独立检索句；不拆可为空")
    final_task: str = Field(description="最后一轮总结的主要任务：直接回答、分述、比较、交集、因果")
