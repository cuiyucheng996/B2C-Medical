"""按 Agent 类型拼 system 提示词。"""

from __future__ import annotations

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

from algorithm.prefix_cache import companion_layered_system_message
from middleware.context import CompanionRunContext
from prompt.companion_system import (
    build_companion_system_dynamic,
    build_companion_system_skill,
    build_companion_system_static,
)

__all__ = ["companion_prompt_middleware", "spell_check_prompt_middleware"]


def _bound_tool_names(request: ModelRequest) -> list[str]:
    names: list[str] = []
    for item in list(getattr(request, "tools", None) or []):
        name = str(getattr(item, "name", None) or getattr(item, "__name__", "") or "").strip()
        if name:
            names.append(name)
    return names


@dynamic_prompt
def companion_prompt_middleware(request: ModelRequest) -> SystemMessage:
    ctx = getattr(request.runtime, "context", None)
    major = intent = handler = ""
    if isinstance(ctx, CompanionRunContext):
        major = str(ctx.major or "")
        intent = str(ctx.intent or "")
        handler = str(ctx.handler or "")
    static = build_companion_system_static()
    skill = build_companion_system_skill(major=major, handler=handler)
    dynamic = build_companion_system_dynamic(
        major=major,
        intent=intent,
        handler=handler,
        bound_tools=_bound_tool_names(request),
    )
    return companion_layered_system_message(static, skill, dynamic)


@dynamic_prompt
def spell_check_prompt_middleware(request: ModelRequest) -> str:
    del request
    from schema.models import CorrectionResult

    return compose_structured_system_prompt(
        load_prompt("spell_check_system"),
        CorrectionResult,
        examples=[
            {
                "original_text": "转氨动偏高",
                "corrected_text": "转氨酶偏高",
                "errors": [],
                "confidence": 0.9,
                "explanation": "专有名词纠错",
                "chunks": [],
            }
        ],
    )
