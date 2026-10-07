"""症状匹配 Agent（LangChain create_agent + lookup_disease_symptom_catalog Tool）。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage

from ...algorithm.tool_trace import extract_tool_trace, summarize_tool_trace
from ...llm.chat import get_chat_model
from ..context.symptom import SymptomRunContext, build_agent_messages, build_run_context
from ..middleware.symptom import SymptomInputMiddleware, SymptomOutputMiddleware
from ..schemas.symptom import SymptomAnalysis, SymptomCheckResult
from ..tools.registry import get_tools_for_sub_agent
from .agent_stream import run_symptom_agent_stream

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "symptom_system_prompt.txt"

__all__ = [
    "analyze_symptom",
    "get_symptom_agent",
]


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache
def get_symptom_agent():
    model = get_chat_model()
    if model is None:
        return None

    tools = get_tools_for_sub_agent("symptom_match")

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_load_system_prompt(),
        middleware=[
            SymptomInputMiddleware(),
            SymptomOutputMiddleware(),
        ],
        response_format=SymptomAnalysis,
        context_schema=SymptomRunContext,
        name="symptom_matcher",
    )


def _analysis_to_result(
    analysis: SymptomAnalysis,
    *,
    used_llm: bool,
    tool_trace: list[dict[str, str]] | None = None,
    rag_context: list[str] | None = None,
) -> SymptomCheckResult:
    normalized = analysis.normalized_description.strip()
    normalized_symptoms = [normalized] if normalized else []
    candidates = list(analysis.candidate_symptoms or [])[:3]
    confirmed = analysis.confirmed_symptom if analysis.matched else None

    return SymptomCheckResult(
        matched=analysis.matched,
        confirmed_symptom=confirmed,
        normalized_symptoms=normalized_symptoms,
        candidate_top3=candidates,
        rag_context=list(rag_context or []),
        reason=analysis.reason or ("LLM 判定" if used_llm else "规则判定"),
        needs_clarification=analysis.needs_clarification,
        is_relevant=not analysis.needs_clarification,
        used_llm=used_llm,
        tool_trace=list(tool_trace or []),
    )


def analyze_symptom(
    state: Mapping[str, Any],
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> SymptomCheckResult:
    """调用 Agent：在已锁定疾病内匹配口语症状。

    ``emit`` 非空时走流式 + ToolCallsAccumulator，拼完整 JSON 后才执行 Tool。
    """
    ctx = build_run_context(state)

    if not ctx.disease_name:
        return SymptomCheckResult(
            matched=False,
            is_relevant=False,
            reason="缺少已锁定疾病，无法匹配症状",
            used_llm=False,
        )

    if not ctx.query:
        return SymptomCheckResult(
            matched=False,
            is_relevant=False,
            needs_clarification=True,
            reason="未收到有效症状描述",
            used_llm=False,
        )

    agent = get_symptom_agent()
    if agent is None:
        raise RuntimeError("LLM not configured")

    try:
        if emit is not None:
            messages, analysis = run_symptom_agent_stream(
                ctx,
                system_prompt=_load_system_prompt(),
                tools=get_tools_for_sub_agent("symptom_match"),
                response_schema=SymptomAnalysis,
                emit=emit,
            )
            result = {"messages": messages, "structured_response": analysis}
        else:
            result = agent.invoke(
                {"messages": build_agent_messages(ctx)},
                context=ctx,
            )
    except Exception:
        logger.exception("symptom agent invoke failed")
        raise

    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError("symptom agent returned no structured_response")

    if isinstance(structured, dict):
        analysis = SymptomAnalysis.model_validate(structured)
    else:
        analysis = structured

    messages: list[BaseMessage] = list(result.get("messages") or [])
    tool_trace = extract_tool_trace(messages, source="thinking_symptom_relevance_check")
    rag_context = summarize_tool_trace(tool_trace)

    if analysis.matched and analysis.confirmed_symptom:
        rag_context.append(f"{analysis.confirmed_symptom}: Agent 高置信命中")

    return _analysis_to_result(
        analysis,
        used_llm=True,
        tool_trace=tool_trace,
        rag_context=rag_context,
    )
