"""工具返回后用第三方模型开思考：data 对不上就提示本轮再调工具。"""

from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from algorithm.tool_think import hint_from_verdict, judge_tool_envelopes, usable_envelopes
from middleware.context import CompanionRunContext

__all__ = ["CompanionThinkMiddleware"]

_MARK = "【思考】"


class ThinkState(AgentState):
    think_count: NotRequired[int]


class CompanionThinkMiddleware(AgentMiddleware):
    state_schema = ThinkState

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        ctx = getattr(runtime, "context", None)
        if not isinstance(ctx, CompanionRunContext) or not ctx.think_enabled:
            return None
        messages = list(state.get("messages") or [])
        if not messages:
            return None
        last = messages[-1]
        if isinstance(last, HumanMessage) and _MARK in str(last.content or ""):
            return None
        envelopes = usable_envelopes(messages)
        if not envelopes:
            return None
        count = int(state.get("think_count") or 0)
        if count >= 2:
            return None
        verdict = judge_tool_envelopes(
            query=ctx.query, major=ctx.major, intent=ctx.intent, envelopes=envelopes
        )
        if verdict is None:
            return None
        nxt = count + 1
        hint = hint_from_verdict(verdict, think_count=nxt)
        return {"think_count": nxt, "messages": [*messages, HumanMessage(content=hint)]}
