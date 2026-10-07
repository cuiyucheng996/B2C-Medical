"""流式 LLM 后端：四种 chunk 经 AgentLoop；全量工具说明书；并行执行。"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool

from src.langchain.agent.loop import LoopContext, run_loop_sync
from src.langchain.tools.registry import ensure_tools_registered

EmitFn = Callable[[dict], None]

__all__ = [
    "build_system_messages",
    "stream_chat_with_tools",
]


def stream_chat_with_tools(
    messages: list[BaseMessage],
    tools: list[BaseTool] | None = None,
    *,
    emit: EmitFn | None = None,
    max_rounds: int = 5,
) -> list[BaseMessage]:
    del max_rounds
    registry = ensure_tools_registered()
    system = ""
    rest: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and not system:
            system = str(msg.content or "")
            continue
        rest.append(msg)
    ctx = LoopContext(system_prompt=system, messages=list(rest))
    bound_tools = list(tools) if tools is not None else registry.langchain_tools_all()
    return run_loop_sync(
        ctx,
        emit=emit,
        registry=registry,
        langchain_tools=bound_tools,
    )


def build_system_messages(system_prompt: str) -> list[BaseMessage]:
    return [SystemMessage(content=system_prompt.strip())]
