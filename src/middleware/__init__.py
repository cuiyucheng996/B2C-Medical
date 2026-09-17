"""C 端 create_agent 中间件栈。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)

from middleware.compress import CompanionCompressMiddleware
from middleware.entity_memory import CompanionEntityMemoryMiddleware
from middleware.envelope import CompanionToolEnvelopeMiddleware
from middleware.mcp_guard import McpGuardMiddleware
from middleware.observe import CompanionObserveMiddleware
from middleware.tool_pairs import CompanionToolPairMiddleware
from middleware.prompt import companion_prompt_middleware, spell_check_prompt_middleware
from middleware.retry_hint import CompanionRetryHintMiddleware
from middleware.spellcheck import SpellCheckBertFirstMiddleware
from middleware.think import CompanionThinkMiddleware
from tools.tool_envelope import dump_envelope

__all__ = ["companion_middleware", "spell_check_middleware"]


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


def companion_middleware() -> list[Any]:
    return [
        companion_prompt_middleware,
        CompanionObserveMiddleware(),
        CompanionEntityMemoryMiddleware(),
        CompanionCompressMiddleware(),
        CompanionRetryHintMiddleware(),
        CompanionThinkMiddleware(),
        CompanionToolPairMiddleware(),
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=6, exit_behavior="continue"),
        McpGuardMiddleware(),
        ToolRetryMiddleware(max_retries=1),
        ToolErrorMiddleware(on_error=_tool_error_content, aon_error=_atool_error_content),
        CompanionToolEnvelopeMiddleware(),
    ]


def spell_check_middleware() -> list[Any]:
    return [
        spell_check_prompt_middleware,
        SpellCheckBertFirstMiddleware(),
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=6, exit_behavior="continue"),
        ToolRetryMiddleware(max_retries=1),
    ]
