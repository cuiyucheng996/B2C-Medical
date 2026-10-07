"""疾病检索子 Agent：第一大轮语义检索候选疾病。"""

from __future__ import annotations

import logging
from typing import Any

from ....algorithm.artifact_store import create_artifact
from ....schemas.agent_protocol import DelegateSubAgentArgs, EmitFn, SubAgentResult, SubAgentTask
from ...tools.tool_entry import ToolMeta
from ...tools.tool_spec import ToolSpec
from ..relevance import analyze_relevance
from .task_state import task_to_state

logger = logging.getLogger(__name__)

__all__ = [
    "DiseaseRetrievalSubAgent",
    "create_disease_retrieval_agent",
]

_DISEASE_RETRIEVAL_TOOL_SPEC = ToolSpec(
    name="disease_retrieval",
    description="第一大轮子 Agent：在已锁定部位下检索候选疾病并返回精炼结论。",
    parameters=DelegateSubAgentArgs,
    visible_to="main",
)

_DISEASE_RETRIEVAL_TOOL_META = ToolMeta(
    title="疾病检索子 Agent",
    summary="委派后执行 RAG + 相关性分析，交回 SubAgentResult。",
    phases=(1,),
    category="subagent",
    operation_hint="主 Agent 第一大轮已选部位、需检索疾病时委派。",
)

_instance: DiseaseRetrievalSubAgent | None = None


class DiseaseRetrievalSubAgent:
    """满足 SubAgent 协议；不继承基类。"""

    @property
    def name(self) -> str:
        return "disease_retrieval"

    @property
    def tool_spec(self) -> ToolSpec:
        return _DISEASE_RETRIEVAL_TOOL_SPEC

    @property
    def tool_meta(self) -> ToolMeta:
        return _DISEASE_RETRIEVAL_TOOL_META

    def _execute(self, task: SubAgentTask, emit: EmitFn | None) -> tuple[SubAgentResult, Any]:
        state = task_to_state(task)
        ctx = task.context or {}
        thread_id = str(ctx.get("thread_id") or "anonymous")

        try:
            raw = analyze_relevance(state, emit=emit)
        except Exception:
            logger.exception("disease_retrieval sub agent failed")
            return SubAgentResult.failed("疾病检索子 Agent 执行失败"), None

        diseases = list(raw.disease_suggested or [])
        extracted = list(raw.extracted_symptoms or [])

        if diseases:
            output = f"检索到候选疾病：{'、'.join(diseases[:3])}"
        elif extracted:
            output = f"提取症状：{'、'.join(extracted[:5])}；{raw.reason}"
        else:
            output = raw.reason or "已完成相关性分析与疾病检索"

        payload = {
            "query": ctx.get("query"),
            "body_part": ctx.get("body_part"),
            "disease_suggested": diseases,
            "extracted_symptoms": extracted,
            "rag_context": list(raw.rag_context or []),
        }
        artifact = create_artifact(
            thread_id=thread_id,
            kind="rag_hits",
            payload=payload,
            summary=output,
        )
        return (
            SubAgentResult.done(
                output,
                artifacts=[artifact],
                context={
                    "extracted_symptoms": extracted,
                    "disease_suggested": diseases,
                },
            ),
            raw,
        )

    def run(self, task: SubAgentTask, emit: EmitFn | None = None) -> SubAgentResult:
        result, _ = self._execute(task, emit)
        return result

    def run_with_raw(
        self,
        task: SubAgentTask,
        emit: EmitFn | None = None,
    ) -> tuple[SubAgentResult, Any]:
        return self._execute(task, emit)


def create_disease_retrieval_agent() -> DiseaseRetrievalSubAgent:
    """工厂：疾病检索子 Agent（进程内单例）。"""
    global _instance
    if _instance is None:
        _instance = DiseaseRetrievalSubAgent()
    return _instance
