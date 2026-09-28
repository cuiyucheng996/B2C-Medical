"""调模型前做增量上下文压缩（已压段落在 Redis session_id 键下，只压新窗口外对话）。"""

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
        session_id = (ctx.session_id or "").strip()
        if not session_id:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        from algorithm.context_compress import apply_incremental_compress

        packed = apply_incremental_compress(
            [msg for msg in messages if isinstance(msg, BaseMessage)],
            session_id=session_id,
        )
        from middleware.middleware_trace import append_middleware_trace

        if len(packed) != len(messages):
            detail = f"增量压缩上下文 {len(messages)} → {len(packed)} 条消息"
        else:
            detail = f"本轮未缩短消息条数（仍为 {len(messages)} 条；未超过保留窗口或摘要未生成）"
        append_middleware_trace(
            "CompanionCompressMiddleware",
            detail,
            runtime=runtime,
        )
        return {"messages": packed}
