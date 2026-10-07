"""症状匹配子 Agent：第二大轮口语 → 官方症状名。"""

from __future__ import annotations

import logging
from typing import Any

from ....algorithm.artifact_store import create_artifact
from ....schemas.agent_protocol import DelegateSubAgentArgs, EmitFn, SubAgentResult, SubAgentTask
from ...tools.tool_entry import ToolMeta
from ...tools.tool_spec import ToolSpec
from ..symptom import analyze_symptom
from .task_state import task_to_state

logger = logging.getLogger(__name__)

__all__ = [
    "SymptomMatchSubAgent",
    "create_symptom_match_agent",
]

_SYMPTOM_MATCH_TOOL_SPEC = ToolSpec(
    name="symptom_match",
    description="第二大轮子 Agent：在已锁定疾病内匹配官方症状名。",
    parameters=DelegateSubAgentArgs,
    visible_to="main",
)

_SYMPTOM_MATCH_TOOL_META = ToolMeta(
    title="症状匹配子 Agent",
    summary="委派后拉取症状表并匹配口语描述，交回 SubAgentResult。",
    phases=(2,),
    category="subagent",
    operation_hint="主 Agent 第二大轮已锁定疾病、需匹配症状时委派。",
)

_instance: SymptomMatchSubAgent | None = None


class SymptomMatchSubAgent:
    @property
    def name(self) -> str:
        return "symptom_match"

    @property
    def tool_spec(self) -> ToolSpec:
        return _SYMPTOM_MATCH_TOOL_SPEC

    @property
    def tool_meta(self) -> ToolMeta:
        return _SYMPTOM_MATCH_TOOL_META

    def _execute(self, task: SubAgentTask, emit: EmitFn | None) -> tuple[SubAgentResult, Any]:
        ctx = task.context or {}
        thread_id = str(ctx.get("thread_id") or "anonymous")
        disease_name = ctx.get("disease_name")
        if not disease_name:
            return SubAgentResult.failed("未指定已锁定疾病，无法匹配症状"), None

        if not str(ctx.get("query") or "").strip():
            snapshot = {
                "sub_agent_id": self.name,
                "thread_id": thread_id,
                "disease_name": disease_name,
            }
            return (
                SubAgentResult.need_input(
                    "能再具体描述一下您的症状吗？例如有什么感觉、持续多久。",
                    snapshot,
                ),
                None,
            )

        state = task_to_state(task)
        try:
            raw = analyze_symptom(state, emit=emit)
        except Exception:
            logger.exception("symptom_match sub agent failed")
            return SubAgentResult.failed("症状匹配子 Agent 执行失败"), None

        normalized = ""
        if raw.normalized_symptoms:
            normalized = str(raw.normalized_symptoms[0] or "")

        if raw.needs_clarification:
            snapshot = {
                "sub_agent_id": self.name,
                "thread_id": thread_id,
                "disease_name": disease_name,
                "candidate_top3": list(raw.candidate_top3 or [])[:3],
            }
            question = raw.reason or "能再具体描述一下症状吗？"
            return (
                SubAgentResult.need_input(
                    question,
                    snapshot,
                    context={
                        "matched": False,
                        "candidate_symptoms": list(raw.candidate_top3 or [])[:3],
                        "normalized_description": normalized,
                    },
                ),
                raw,
            )

        if raw.matched and raw.confirmed_symptom:
            output = f"已匹配官方症状「{raw.confirmed_symptom}」"
        elif raw.candidate_top3:
            output = f"未精确命中，候选：{'、'.join(raw.candidate_top3[:3])}"
        else:
            output = raw.reason or "已完成症状匹配分析"

        payload = {
            "disease_name": disease_name,
            "query": ctx.get("query"),
            "matched": raw.matched,
            "confirmed_symptom": raw.confirmed_symptom,
            "candidate_top3": list(raw.candidate_top3 or [])[:3],
            "normalized_symptoms": list(raw.normalized_symptoms or []),
            "rag_context": list(raw.rag_context or []),
        }
        artifact = create_artifact(
            thread_id=thread_id,
            kind="symptom_catalog",
            payload=payload,
            summary=output,
        )
        return (
            SubAgentResult.done(
                output,
                artifacts=[artifact],
                context={
                    "matched": raw.matched,
                    "confirmed_symptom": raw.confirmed_symptom,
                    "candidate_symptoms": list(raw.candidate_top3 or [])[:3],
                    "normalized_description": normalized,
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


def create_symptom_match_agent() -> SymptomMatchSubAgent:
    """工厂：症状匹配子 Agent（进程内单例）。"""
    global _instance
    if _instance is None:
        _instance = SymptomMatchSubAgent()
    return _instance
