"""纠错 Agent：提醒先调 BERT。"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, ToolMessage

__all__ = ["SpellCheckBertFirstMiddleware"]

_HINT = "【调度】尚未调用工具。必须先调用 bert_spell_correct，再视情况用 check_common_errors / query_dictionary。"


class SpellCheckBertFirstMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        del runtime
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        if any(isinstance(msg, ToolMessage) for msg in messages):
            return None
        last = messages[-1]
        if not isinstance(last, HumanMessage):
            return None
        if _HINT in str(last.content or ""):
            return None
        patched = HumanMessage(content=f"{last.content}\n\n{_HINT}")
        return {"messages": [*messages[:-1], patched]}
