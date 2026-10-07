"""相关性 Agent 中间件：入参整理、出参校验与日志。"""

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

from ..context.relevance import RelevanceRunContext
from ..schemas.relevance import RelevanceAnalysis

logger = logging.getLogger(__name__)

__all__ = [
    "RelevanceInputMiddleware",
    "RelevanceOutputMiddleware",
]


class RelevanceInputMiddleware(AgentMiddleware[AgentState[Any], RelevanceRunContext]):
    """入参中间件：按是否已有部位补充 Agent 行动提示。"""

    def before_agent(
        self,
        state: AgentState[Any],
        runtime: Any,
    ) -> dict[str, Any] | None:
        ctx: RelevanceRunContext | None = getattr(runtime, "context", None)
        if ctx is None:
            return None

        if not ctx.query:
            logger.warning("relevance agent received empty query")
        return None

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx: RelevanceRunContext | None = getattr(runtime, "context", None)
        if ctx is None:
            return None

        messages = list(state.get("messages") or [])
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None

        if ctx.existing_body_part:
            suffix = (
                f"\n\n【注意】用户已通过选项选定部位「{ctx.existing_body_part}」。"
                "若输入与症状相关，请调用 search_diseases_by_symptom 检索候选疾病；"
                "query 传用户原文，body_part 传该部位；结构化输出中 body_part 保持 null。"
            )
        else:
            suffix = (
                "\n\n【注意】当前无已选部位。"
                "若输入与症状相关，请从八大部位推断 body_part；"
                "不要调用 search_diseases_by_symptom。"
            )

        if suffix not in str(last.content):
            patched = HumanMessage(content=f"{last.content}{suffix}")
            return {"messages": [*messages[:-1], patched]}
        return None


class RelevanceOutputMiddleware(AgentMiddleware[AgentState[Any], RelevanceRunContext]):
    """出参中间件：记录分析结果，校验 body_part 合法性。"""

    def after_agent(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        structured = state.get("structured_response")
        if structured is None:
            logger.warning("relevance agent finished without structured_response")
            return None

        if isinstance(structured, dict):
            analysis = RelevanceAnalysis.model_validate(structured)
        elif isinstance(structured, RelevanceAnalysis):
            analysis = structured
        else:
            logger.warning("unexpected structured_response type: %s", type(structured))
            return None

        ctx: RelevanceRunContext | None = getattr(runtime, "context", None)
        if ctx and ctx.existing_body_part:
            analysis.body_part = None

        logger.info(
            "relevance agent result: relevant=%s chitchat=%s sensitive=%s body=%s reason=%s",
            analysis.is_relevant,
            analysis.is_chitchat,
            analysis.is_sensitive,
            analysis.body_part,
            analysis.reason,
        )
        return {"structured_response": analysis}

    def wrap_model_call(
        self,
        request: ModelRequest[RelevanceRunContext],
        handler: Any,
    ) -> ModelResponse[Any]:
        response = handler(request)
        if response.structured_response is not None:
            logger.debug("relevance model structured output received")
        return response
