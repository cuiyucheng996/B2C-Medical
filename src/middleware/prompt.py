"""按 Agent 类型拼 system 提示词。"""

from __future__ import annotations

from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest

from middleware.context import CompanionRunContext
from prompt import load_prompt

__all__ = ["companion_prompt_middleware", "spell_check_prompt_middleware"]


@dynamic_prompt
def companion_prompt_middleware(request: ModelRequest) -> str:
    ctx = getattr(request.runtime, "context", None)
    parts = [load_prompt("companion_tools"), load_prompt("companion_compose")]
    if isinstance(ctx, CompanionRunContext) and (ctx.major or ctx.intent or ctx.handler):
        parts.append(
            f"本任务只处理：{ctx.major} / {ctx.intent}（{ctx.handler}）。并行工具结果请按此甄别。"
        )
    return "\n\n".join(parts)


@dynamic_prompt
def spell_check_prompt_middleware(request: ModelRequest) -> str:
    del request
    return load_prompt("spell_check_system")
