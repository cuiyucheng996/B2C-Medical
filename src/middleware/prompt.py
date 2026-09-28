"""按 Agent 类型拼 system 提示词。"""

from __future__ import annotations

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

from middleware.context import CompanionRunContext
from prompt import load_prompt
from prompt.companion_dispatch import build_companion_tools_prompt

__all__ = ["companion_prompt_middleware", "spell_check_prompt_middleware"]


def _bound_tool_names(request: ModelRequest) -> list[str]:
    names: list[str] = []
    for item in list(getattr(request, "tools", None) or []):
        name = str(getattr(item, "name", None) or getattr(item, "__name__", "") or "").strip()
        if name:
            names.append(name)
    return names


@dynamic_prompt
def companion_prompt_middleware(request: ModelRequest) -> str:
    ctx = getattr(request.runtime, "context", None)
    major = intent = handler = ""
    if isinstance(ctx, CompanionRunContext):
        major = str(ctx.major or "")
        intent = str(ctx.intent or "")
        handler = str(ctx.handler or "")
    parts = [
        build_companion_tools_prompt(
            major=major,
            intent=intent,
            handler=handler,
            bound_tools=_bound_tool_names(request),
        ),
        load_prompt("companion_compose"),
    ]
    if major or intent or handler:
        parts.append(
            f"本任务只处理：{major} / {intent}（{handler}）。勿回答其它意图；并行工具结果按此甄别。"
        )
    return "\n\n".join(parts)


@dynamic_prompt
def spell_check_prompt_middleware(request: ModelRequest) -> str:
    del request
    return load_prompt("spell_check_system")
