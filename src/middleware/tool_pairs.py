"""发给模型前丢掉对不上的 ToolMessage，避免 OpenAI 兼容口 400。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage

__all__ = ["CompanionToolPairMiddleware", "drop_orphan_tool_messages"]


def _call_ids(msg: AIMessage) -> set[str]:
    ids: set[str] = set()
    for item in getattr(msg, "tool_calls", None) or []:
        if isinstance(item, dict):
            tid = str(item.get("id") or "")
        else:
            tid = str(getattr(item, "id", "") or "")
        if tid:
            ids.add(tid)
    return ids


def drop_orphan_tool_messages(messages: list[Any]) -> list[Any]:
    out: list[Any] = []
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            pending = _call_ids(msg)
            out.append(msg)
            continue
        if isinstance(msg, ToolMessage):
            tid = str(getattr(msg, "tool_call_id", "") or "")
            if tid and tid in pending:
                out.append(msg)
                pending.discard(tid)
            continue
        pending = set()
        out.append(msg)
    return out


class CompanionToolPairMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        packed = drop_orphan_tool_messages(messages)
        if packed == messages:
            return None
        return {"messages": packed}
