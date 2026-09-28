"""C 端 create_agent 中间件栈。"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)

from middleware.hitl_config import companion_tool_hitl_enabled, companion_tool_interrupt_on

from middleware.compress import CompanionCompressMiddleware
from middleware.entity_memory import CompanionEntityMemoryMiddleware
from middleware.envelope import CompanionToolEnvelopeMiddleware
from middleware.mcp_guard import McpGuardMiddleware
from middleware.observe import CompanionObserveMiddleware
from middleware.tool_trace import CompanionToolTraceMiddleware
from middleware.tool_pairs import CompanionToolPairMiddleware
from middleware.prompt import companion_prompt_middleware, spell_check_prompt_middleware
from middleware.retry_hint import CompanionRetryHintMiddleware
from middleware.spellcheck import SpellCheckBertFirstMiddleware
from middleware.think import CompanionThinkMiddleware
from middleware.llm_trace import CompanionLlmTraceMiddleware
from tools.tool_envelope import dump_envelope

__all__ = [
    "companion_middleware",
    "companion_model_run_limit",
    "companion_tool_run_limit",
    "spell_check_middleware",
]


def companion_model_run_limit() -> int:
    """单路 create_agent 一次 ainvoke 内最多几次 model（默认 3：定工具 + 可能重试/思考 + 组句）。"""
    raw = (os.getenv("COMPANION_MODEL_RUN_LIMIT") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(n, 8))


def companion_tool_run_limit() -> int:
    """与 model 次数配套，默认单轮最多 2 次 tool（含并行多条）。"""
    raw = (os.getenv("COMPANION_TOOL_RUN_LIMIT") or "2").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 2
    return max(1, min(n, 8))


def _tool_error_content(exc: Exception, request: Any) -> str:
    name = str((getattr(request, "tool_call", None) or {}).get("name") or "")
    return dump_envelope(
        tool=name,
        ok=False,
        error=type(exc).__name__,
        usable=False,
        retryable=True,
    )


async def _atool_error_content(exc: Exception, request: Any) -> str:
    return _tool_error_content(exc, request)


def companion_middleware(*, enable_tool_hitl: bool = True) -> list[Any]:
    stack: list[Any] = [
        companion_prompt_middleware,
        CompanionLlmTraceMiddleware(),
        CompanionObserveMiddleware(),
        CompanionEntityMemoryMiddleware(),
        CompanionCompressMiddleware(),
        CompanionRetryHintMiddleware(),
        CompanionThinkMiddleware(),
        CompanionToolPairMiddleware(),
        ModelCallLimitMiddleware(
            run_limit=companion_model_run_limit(),
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            run_limit=companion_tool_run_limit(),
            exit_behavior="continue",
        ),
        McpGuardMiddleware(),
        ToolRetryMiddleware(max_retries=1),
        ToolErrorMiddleware(on_error=_tool_error_content, aon_error=_atool_error_content),
        CompanionToolEnvelopeMiddleware(),
        CompanionToolTraceMiddleware(),
    ]
    if enable_tool_hitl and companion_tool_hitl_enabled():
        stack.insert(6, HumanInTheLoopMiddleware(interrupt_on=companion_tool_interrupt_on()))
    return stack


def spell_check_middleware() -> list[Any]:
    return [
        spell_check_prompt_middleware,
        CompanionLlmTraceMiddleware(fixed_stage="纠错 Agent"),
        SpellCheckBertFirstMiddleware(),
        ModelCallLimitMiddleware(
            run_limit=companion_model_run_limit(),
            exit_behavior="end",
        ),
        ToolCallLimitMiddleware(
            run_limit=companion_tool_run_limit(),
            exit_behavior="continue",
        ),
        ToolRetryMiddleware(max_retries=1),
    ]
