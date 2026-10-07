"""Agent 流式执行：tool_calls 经 llm_backend 累积后再执行，最后 structured output。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from ..context.relevance import RelevanceRunContext, build_agent_messages
from ..context.symptom import SymptomRunContext

logger = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], None]
SchemaT = TypeVar("SchemaT", bound=BaseModel)

__all__ = [
    "run_structured_agent_invoke",
    "run_structured_agent_stream",
]


def _apply_relevance_suffix(messages: list[BaseMessage], ctx: RelevanceRunContext) -> list[BaseMessage]:
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return messages
    if ctx.existing_body_part:
        suffix = (
            f"\n\n【注意】用户已通过选项选定部位「{ctx.existing_body_part}」。"
            "若输入与症状相关，请调用 search_diseases_by_symptom 检索候选疾病；"
            "query 传用户原文，body_part 传该部位；结构化输出中 body_part 保持 null。"
        )
    else:
        suffix = (
            "\n\n【注意】当前无已选部位。"
            "若输入与症状相关，请推断 body_part，并调用 search_diseases_by_symptom；"
            "query 传用户原文，body_part 传推断部位。"
        )
    if suffix in str(last.content):
        return messages
    return [*messages[:-1], HumanMessage(content=f"{last.content}{suffix}")]


def _apply_symptom_suffix(messages: list[BaseMessage], ctx: SymptomRunContext) -> list[BaseMessage]:
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return messages
    suffix = (
        f"\n\n【注意】已锁定疾病为「{ctx.disease_name or '（缺失）'}」。"
        "请先调用 lookup_disease_symptom_catalog 获取官方症状列表，再判断 matched / candidate_symptoms。"
    )
    if suffix in str(last.content):
        return messages
    return [*messages[:-1], HumanMessage(content=f"{last.content}{suffix}")]


def _invoke_structured_output(
    model: Any,
    response_schema: type[SchemaT],
    messages: list[BaseMessage],
) -> SchemaT:
    """DeepSeek 等不支持 json_schema response_format，优先 function_calling。"""
    last_error: Exception | None = None
    for method in ("function_calling", "json_mode"):
        try:
            structured_model = model.with_structured_output(
                response_schema,
                method=method,
            )
            analysis = structured_model.invoke(messages)
            if isinstance(analysis, dict):
                analysis = response_schema.model_validate(analysis)
            return analysis
        except Exception as exc:
            last_error = exc
            logger.warning("structured output method=%s failed: %s", method, exc)
    if last_error is not None:
        raise last_error
    raise RuntimeError("structured output 未返回结果")


def run_structured_agent_invoke(
    *,
    system_prompt: str,
    agent_messages: list[BaseMessage],
    response_schema: type[SchemaT],
    emit: EmitFn | None = None,
) -> SchemaT:
    """无 Tool 的结构化调用（主 Agent 计划 / 答复环节）。"""
    from src.langchain.agent.stream_agent import chunk_text
    from src.llm.chat import get_chat_model
    from src.llm.llm_backend import build_system_messages

    model = get_chat_model()
    if model is None:
        raise RuntimeError("LLM not configured")

    messages = build_system_messages(system_prompt) + list(agent_messages)
    if emit is not None:
        for chunk in model.stream(messages):
            text = chunk_text(chunk.content)
            if text:
                emit({"event": "token", "content": text})

    analysis = _invoke_structured_output(model, response_schema, messages)
    return analysis


def run_structured_agent_stream(
    *,
    system_prompt: str,
    agent_messages: list[BaseMessage],
    tools: list[Any],
    response_schema: type[SchemaT],
    emit: EmitFn | None,
    suffix_fn: Callable[[list[BaseMessage], Any], list[BaseMessage]] | None = None,
    context: Any | None = None,
) -> tuple[list[BaseMessage], SchemaT]:
    """流式 Tool 循环 + 最终 structured output（不经过 create_agent.invoke）。"""
    from src.llm.chat import get_chat_model
    from src.llm.llm_backend import build_system_messages, stream_chat_with_tools

    model = get_chat_model()
    if model is None:
        raise RuntimeError("LLM not configured")

    messages = build_system_messages(system_prompt) + list(agent_messages)
    if suffix_fn is not None and context is not None:
        messages = build_system_messages(system_prompt) + suffix_fn(list(agent_messages), context)

    transcript = stream_chat_with_tools(messages, tools, emit=emit)
    analysis = _invoke_structured_output(model, response_schema, transcript)
    return transcript, analysis


def run_relevance_agent_stream(
    ctx: RelevanceRunContext,
    *,
    system_prompt: str,
    tools: list[Any],
    response_schema: type[SchemaT],
    emit: EmitFn | None,
) -> tuple[list[BaseMessage], SchemaT]:
    return run_structured_agent_stream(
        system_prompt=system_prompt,
        agent_messages=build_agent_messages(ctx),
        tools=tools,
        response_schema=response_schema,
        emit=emit,
        suffix_fn=_apply_relevance_suffix,
        context=ctx,
    )


def run_symptom_agent_stream(
    ctx: SymptomRunContext,
    *,
    system_prompt: str,
    tools: list[Any],
    response_schema: type[SchemaT],
    emit: EmitFn | None,
) -> tuple[list[BaseMessage], SchemaT]:
    from ..context.symptom import build_agent_messages as build_symptom_messages

    return run_structured_agent_stream(
        system_prompt=system_prompt,
        agent_messages=build_symptom_messages(ctx),
        tools=tools,
        response_schema=response_schema,
        emit=emit,
        suffix_fn=_apply_symptom_suffix,
        context=ctx,
    )
