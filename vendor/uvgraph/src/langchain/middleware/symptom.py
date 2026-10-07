"""症状匹配 Agent 中间件。"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

from ..context.symptom import SymptomRunContext
from ..schemas.symptom import SymptomAnalysis

logger = logging.getLogger(__name__)

__all__ = [
    "SymptomInputMiddleware",
    "SymptomOutputMiddleware",
]


class SymptomInputMiddleware(AgentMiddleware[AgentState[Any], SymptomRunContext]):
    """入参中间件：提示必须先查疾病详情再匹配症状。"""

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx: SymptomRunContext | None = getattr(runtime, "context", None)
        if ctx is None:
            return None

        messages = list(state.get("messages") or [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None

        suffix = (
            f"\n\n【注意】已锁定疾病为「{ctx.disease_name or '（缺失）'}」。"
            "请先调用 lookup_disease_symptom_catalog 获取官方症状列表，再判断 matched / candidate_symptoms。"
        )
        if suffix not in str(last.content):
            patched = HumanMessage(content=f"{last.content}{suffix}")
            return {"messages": [*messages[:-1], patched]}
        return None


class SymptomOutputMiddleware(AgentMiddleware[AgentState[Any], SymptomRunContext]):
    """出参中间件：记录结构化结果。"""

    def after_agent(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        structured = state.get("structured_response")
        if structured is None:
            logger.warning("symptom agent finished without structured_response")
            return None

        if isinstance(structured, dict):
            analysis = SymptomAnalysis.model_validate(structured)
        elif isinstance(structured, SymptomAnalysis):
            analysis = structured
        else:
            logger.warning("unexpected symptom structured_response type: %s", type(structured))
            return None

        logger.info(
            "symptom agent result: matched=%s confirmed=%s candidates=%s reason=%s",
            analysis.matched,
            analysis.confirmed_symptom,
            analysis.candidate_symptoms,
            analysis.reason,
        )
        return {"structured_response": analysis}

    def wrap_model_call(
        self,
        request: ModelRequest[SymptomRunContext],
        handler: Any,
    ) -> ModelResponse[Any]:
        response = handler(request)
        if response.structured_response is not None:
            logger.debug("symptom model structured output received")
        return response
