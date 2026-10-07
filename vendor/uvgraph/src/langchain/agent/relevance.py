"""相关性分析 Agent（LangChain create_agent + DeepSeek + RAG Tools）。"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, ToolMessage

from ...algorithm.tool_trace import extract_tool_trace, summarize_tool_trace
from ...llm.chat import get_chat_model
from ..context.relevance import RelevanceRunContext, build_agent_messages, build_run_context
from ..middleware.relevance import RelevanceInputMiddleware, RelevanceOutputMiddleware
from ..schemas.relevance import RelevanceAnalysis, RelevanceResult
from ..tools.medical_utils import search_diseases_in_milvus
from ..tools.registry import get_tools_for_sub_agent
from .agent_stream import run_relevance_agent_stream

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "relevance_system_prompt.txt"
_AGENT: Any | None = None

__all__ = [
    "analyze_relevance",
    "get_relevance_agent",
]


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()


@lru_cache
def get_relevance_agent():
    model = get_chat_model()
    if model is None:
        return None

    tools = get_tools_for_sub_agent("disease_retrieval")

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=_load_system_prompt(),
        middleware=[
            RelevanceInputMiddleware(),
            RelevanceOutputMiddleware(),
        ],
        #告诉 LangChain：最后让 LLM 按 RelevanceAnalysis 这个 schema 填结果。
        response_format=RelevanceAnalysis,
        #告诉 LangChain：最后让 LLM 按 RelevanceRunContext 这个 schema 填结果。
        context_schema=RelevanceRunContext,
        name="relevance_checker",
    )


def _analysis_to_result(
    analysis: RelevanceAnalysis,
    *,
    used_llm: bool,
    existing_body_part: str | None,
    disease_suggested: list[str] | None = None,
    rag_context: list[str] | None = None,
    tool_trace: list[dict[str, str]] | None = None,
) -> RelevanceResult:
    is_chitchat = analysis.is_chitchat
    is_relevant = analysis.is_relevant and not analysis.is_sensitive and not is_chitchat
    body_part = existing_body_part or (analysis.body_part if is_relevant else None)
    return RelevanceResult(
        is_relevant=is_relevant,
        is_sensitive=analysis.is_sensitive,
        is_chitchat=is_chitchat,
        body_part=body_part if is_relevant else None,  # type: ignore[arg-type]
        reason=analysis.reason or ("LLM 判定" if used_llm else "规则判定"),
        extracted_symptoms=list(analysis.extracted_symptoms),
        used_llm=used_llm,
        disease_suggested=list(disease_suggested or []),
        rag_context=list(rag_context or []),
        tool_trace=list(tool_trace or []),
    )


def _parse_tool_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, str):
        text = content.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _extract_rag_hits(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name not in {"search_diseases_by_symptom", "search_diseases_rag"}:
            continue
        hits = _parse_tool_content(msg.content)
        if hits:
            break
    return hits


def _disease_fields_from_hits(hits: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    suggested: list[str] = []
    rag_context: list[str] = []
    for hit in hits[:3]:
        disease = str(hit.get("disease") or "").strip()
        if not disease or disease in suggested:
            continue
        suggested.append(disease)
        score = hit.get("score")
        detail = str(hit.get("symptoms") or hit.get("synonyms") or "")
        if score is not None:
            rag_context.append(f"{disease}({float(score):.3f}): {detail}")
        else:
            rag_context.append(f"{disease}: {detail}")
    return suggested, rag_context


def analyze_relevance(
    state: Mapping[str, Any],
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> RelevanceResult:
    """调用 Agent：无部位时推断部位；有部位时 Tool 检索疾病。

    ``emit`` 非空时走流式：token / tool_call_fragment 直推前端，Tool 仅在累积器拼完整后执行。
    """
    ctx = build_run_context(state)

    agent = get_relevance_agent()
    if agent is None:
        raise RuntimeError("LLM not configured")

    if not ctx.query:
        return RelevanceResult(
            is_relevant=False,
            is_chitchat=True,
            reason="描述过短或过于模糊",
            used_llm=False,
        )

    try:
        if emit is not None:
            messages, analysis = run_relevance_agent_stream(
                ctx,
                system_prompt=_load_system_prompt(),
                tools=get_tools_for_sub_agent("disease_retrieval"),
                response_schema=RelevanceAnalysis,
                emit=emit,
            )
            result = {"messages": messages, "structured_response": analysis}
        else:
            result = agent.invoke(
                {"messages": build_agent_messages(ctx)},
                context=ctx,
            )
    except Exception:
        logger.exception("relevance agent invoke failed")
        raise

    #structured_response就是对应这个response_format呗
    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError("relevance agent returned no structured_response")

    if isinstance(structured, dict):
        analysis = RelevanceAnalysis.model_validate(structured)
    else:
        analysis = structured

    if ctx.existing_body_part:
        analysis.body_part = None

    messages = list(result.get("messages") or [])
    tool_trace = extract_tool_trace(messages, source="thinking_relevance_check")
    hits = _extract_rag_hits(messages)
    # Agent 未调 Tool 时，有已选部位则子进程 RAG 兜底
    if (
        not hits
        and ctx.existing_body_part
        and analysis.is_relevant
        and not analysis.is_sensitive
        and not analysis.is_chitchat
    ):
        hits = search_diseases_in_milvus(
            ctx.query,
            body_part=ctx.existing_body_part,
            limit=5,
        )

    disease_suggested, rag_context = _disease_fields_from_hits(hits)
    if not rag_context:
        rag_context = summarize_tool_trace(tool_trace)

    return _analysis_to_result(
        analysis,
        used_llm=True,
        existing_body_part=ctx.existing_body_part,
        disease_suggested=disease_suggested,
        rag_context=rag_context,
        tool_trace=tool_trace,
    )
