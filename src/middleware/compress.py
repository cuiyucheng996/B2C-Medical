"""调模型前做增量上下文压缩（已压段落在 Redis，只压新窗口外对话）。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import BaseMessage

from middleware.context import CompanionRunContext

__all__ = ["CompanionCompressMiddleware"]


class CompanionCompressMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx = getattr(runtime, "context", None)
        if not isinstance(ctx, CompanionRunContext):
            return None
        user_id = (ctx.user_id or "").strip()
        if not user_id:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        from algorithm.context_compress import apply_incremental_compress

        packed = apply_incremental_compress(
            [msg for msg in messages if isinstance(msg, BaseMessage)],
            user_id=user_id,
            session_id=ctx.session_id,
        )
        return {"messages": packed}
