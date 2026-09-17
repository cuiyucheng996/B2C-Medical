"""工具信封显示证据不足时，提醒模型最多再调一次。"""

from __future__ import annotations

import json
from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

__all__ = ["CompanionRetryHintMiddleware"]

_HINT = (
    "【工具甄别】最近工具信封 usable=false 且 retryable=true。"
    "可换参数再调同一工具一次；若仍空则停止调工具并如实说查不到。"
)


class RetryHintState(AgentState):
    retry_hinted: NotRequired[bool]


class CompanionRetryHintMiddleware(AgentMiddleware):
    state_schema = RetryHintState

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        del runtime
        if state.get("retry_hinted"):
            return None
        messages = list(state.get("messages") or [])
        tails: list[ToolMessage] = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tails.append(msg)
                continue
            break
        if not tails:
            return None
        if not all(_retryable_empty(msg) for msg in tails):
            return None
        return {
            "retry_hinted": True,
            "messages": [*messages, HumanMessage(content=_HINT)],
        }


def _retryable_empty(msg: ToolMessage) -> bool:
    try:
        payload = json.loads(str(msg.content or ""))
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("retryable") is True and payload.get("usable") is False
