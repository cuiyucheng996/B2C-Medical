"""模型调用前：用 ConversationEntityMemory + ConversationKGMemory 抽实体/三元组并注入。"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from algorithm.entity_kg_memory import recall_and_update
from middleware.context import CompanionRunContext

__all__ = ["CompanionEntityMemoryMiddleware"]

_MARK = "【实体记忆】"


class CompanionEntityMemoryMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx = getattr(runtime, "context", None)
        if not isinstance(ctx, CompanionRunContext):
            return None
        query = (ctx.query or "").strip()
        if not query:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        if any(
            isinstance(msg, HumanMessage) and _MARK in str(getattr(msg, "content", "") or "")
            for msg in messages
        ):
            return None
        note = recall_and_update(
            query,
            messages,
            user_key=ctx.user_id or ctx.session_id,
        )
        if not note.strip():
            return None
        from middleware.middleware_trace import append_middleware_trace

        append_middleware_trace(
            "CompanionEntityMemoryMiddleware",
            f"注入实体/图谱记忆（{len(note)} 字）",
            runtime=runtime,
        )
        return {"messages": [*messages, HumanMessage(content=note)]}

    async def abefore_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.before_model, state, runtime)
