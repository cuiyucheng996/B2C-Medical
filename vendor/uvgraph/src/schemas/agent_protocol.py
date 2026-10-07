"""主/子 Agent 委派协议（纯类型，无 LangGraph 依赖）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .consult import BodyPart

__all__ = [
    "Artifact",
    "EmitFn",
    "MainAgentPlan",
    "SubAgent",
    "SubAgentConclusion",
    "SubAgentDelegation",
    "DelegateSubAgentArgs",
    "SubAgentId",
    "SubAgentResult",
    "SubAgentResultStatus",
    "SubAgentStatus",
    "SubAgentTask",
    "consult_fields_from_result",
]

SubAgentId = Literal["disease_retrieval", "symptom_match", "none"]
SubAgentResultStatus = Literal["done", "need_input", "failed"]
# 兼容旧代码：委派跳过 / 调度错误
SubAgentStatus = Literal["success", "skipped", "error"]

EmitFn = Callable[[dict[str, Any]], None]


class Artifact(BaseModel):
    """大对象取件码：summary 进 LLM，ref 跨轮传递，download_url 供前端下载。"""

    ref: str = Field(description="artifact://medical/{thread_id}/{id}/{kind}")
    kind: str = Field(description="产物类型，如 rag_hits / symptom_catalog / report")
    summary: str = Field(default="", description="一句话摘要（进 LLM 上下文）")
    size_bytes: int = Field(default=0, description="产物字节大小")
    download_url: str | None = Field(default=None, description="前端 REST 下载端点")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "summary": self.summary,
            "size_bytes": self.size_bytes,
            "download_url": self.download_url,
        }


class SubAgentTask(BaseModel):
    """主 Agent 下发给子 Agent 的任务单（自包含，无需对话历史）。"""

    instruction: str = Field(
        description="自然语言任务描述（含指代解析后的明确信息）",
    )
    refs: list[str] = Field(
        default_factory=list,
        description="相关 artifact 句柄",
    )
    resume_token: str | None = Field(
        default=None,
        description=(
            "子 Agent / Tool 中途等人：SessionStore 快照钥匙；首次为 None。"
            "与问诊图 thread_id 不是同一把钥匙。"
        ),
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="额外上下文（major_round、body_part、tools_hint 等）",
    )


class DelegateSubAgentArgs(BaseModel):
    """主 Agent 委派子 Agent 时的 OpenAI tool 参数（与 SubAgentTask 对齐）。"""

    instruction: str = Field(description="自包含任务描述")
    refs: list[str] = Field(default_factory=list, description="可引用的 artifact ref")
    resume_token: str | None = Field(default=None, description="挂起恢复 token")


class SubAgentResult(BaseModel):
    """子 Agent 交回主 Agent 的结果单（三态约定）。"""

    status: SubAgentResultStatus
    output: str = Field(default="", description="给主 Agent 的精炼结论")
    question: str | None = Field(
        default=None,
        description="status=need_input 时对用户的追问",
    )
    snapshot: dict[str, Any] | None = Field(
        default=None,
        description="挂起时的状态快照（供 resume_token 恢复）",
    )
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = Field(default=None, description="status=failed 时的错误信息")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="问诊业务字段兜底口袋（disease_suggested、matched 等）",
    )

    @classmethod
    def done(
        cls,
        output: str,
        *,
        artifacts: list[Artifact] | None = None,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        return cls(
            status="done",
            output=output,
            artifacts=artifacts or [],
            context=context or {},
        )

    @classmethod
    def need_input(
        cls,
        question: str,
        snapshot: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        return cls(
            status="need_input",
            output=question,
            question=question,
            snapshot=snapshot,
            context=context or {},
        )

    @classmethod
    def failed(cls, error: str) -> SubAgentResult:
        return cls(status="failed", error=error, output=error)


class SubAgentDelegation(BaseModel):
    """主 Agent 计划中的委派意图：派谁 + 工具 hint + 任务摘要。"""

    sub_agent_id: SubAgentId = Field(
        description="子 Agent 标识；无需派发时填 none",
    )
    tools_hint: list[str] = Field(
        default_factory=list,
        description="建议子 Agent 优先调用的工具名",
    )
    task_brief: str = Field(
        default="",
        description="任务摘要，将展开为 SubAgentTask.instruction",
    )


class MainAgentPlan(BaseModel):
    """主 Agent 计划：听懂意图 + 路由判定 + 委派，不含 Tool 执行结果。"""

    intent: str = Field(description="用户本轮意图摘要")
    is_relevant: bool = Field(description="是否与当前问诊阶段任务相关")
    is_chitchat: bool = Field(default=False, description="是否纯寒暄/无关话题")
    is_sensitive: bool = Field(default=False, description="是否敏感违规内容")
    needs_clarification: bool = Field(default=False, description="是否描述过短模糊需追问")
    reason: str = Field(default="", description="简短判定理由")
    body_part: BodyPart | None = Field(
        default=None,
        description="第一大轮：无已选部位时推断部位；已有部位时保持 null",
    )
    delegation: SubAgentDelegation | None = Field(
        default=None,
        description="需子 Agent 执行脏活时填写；否则 null 或 sub_agent_id=none",
    )


class SubAgentConclusion(BaseModel):
    """兼容层：从 SubAgentResult 解包问诊字段，供图节点 / 旧代码消费。"""

    sub_agent_id: SubAgentId
    status: SubAgentStatus = "success"
    conclusion: str = Field(default="", description="精炼结论")
    extracted_symptoms: list[str] = Field(default_factory=list)
    disease_suggested: list[str] = Field(default_factory=list)
    matched: bool = False
    confirmed_symptom: str | None = None
    candidate_symptoms: list[str] = Field(default_factory=list)
    normalized_description: str = ""

    @classmethod
    def from_result(cls, sub_agent_id: SubAgentId, result: SubAgentResult) -> SubAgentConclusion:
        fields = consult_fields_from_result(result)
        if result.status == "failed":
            status: SubAgentStatus = "error"
        elif sub_agent_id == "none":
            status = "skipped"
        else:
            status = "success"
        return cls(
            sub_agent_id=sub_agent_id,
            status=status,
            conclusion=result.output,
            **fields,
        )


def consult_fields_from_result(result: SubAgentResult) -> dict[str, Any]:
    """从 SubAgentResult.context 解包问诊业务字段。"""
    ctx = result.context or {}
    return {
        "extracted_symptoms": list(ctx.get("extracted_symptoms") or []),
        "disease_suggested": list(ctx.get("disease_suggested") or []),
        "matched": bool(ctx.get("matched")),
        "confirmed_symptom": ctx.get("confirmed_symptom"),
        "candidate_symptoms": list(ctx.get("candidate_symptoms") or [])[:3],
        "normalized_description": str(ctx.get("normalized_description") or ""),
    }


@runtime_checkable
class SubAgent(Protocol):
    """子 Agent 实现协议：有 name + run 即为子 Agent（结构化子类型）。"""

    @property
    def name(self) -> str: ...

    def run(
        self,
        task: SubAgentTask,
        emit: EmitFn | None = None,
    ) -> SubAgentResult: ...
