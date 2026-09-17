"""把预处理观察补进本轮 Human，避免节点里手拼 system。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage

from middleware.context import CompanionRunContext

__all__ = ["CompanionObserveMiddleware"]


class CompanionObserveMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx = getattr(runtime, "context", None)
        if not isinstance(ctx, CompanionRunContext):
            return None
        observe = (ctx.observe or "").strip()
        if not observe:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        marker = "【预处理观察】"
        last = messages[-1]
        if isinstance(last, HumanMessage) and marker in str(last.content or ""):
            return None
        if any(isinstance(msg, HumanMessage) and marker in str(getattr(msg, "content", "") or "") for msg in messages):
            return None
        extra = HumanMessage(content=f"{marker}\n{observe}\n请按需调用工具后回复用户。")
        return {"messages": [*messages, extra]}
