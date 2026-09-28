"""create_agent 每次模型调用：记录送入模型的完整 messages（含 system）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from algorithm.llm_trace import (
    STAGE_AGENT,
    note_token_usage,
    record_llm_call,
    serialize_langchain_messages,
    take_llm_trace,
)
from middleware.context import CompanionRunContext

__all__ = ["CompanionLlmTraceMiddleware"]


class CompanionLlmTraceMiddleware(AgentMiddleware):
    def __init__(self, *, fixed_stage: str = "") -> None:
        self._fixed_stage = (fixed_stage or "").strip()

    def _stage(self) -> str:
        return self._fixed_stage or STAGE_AGENT

    def _route_label(self, runtime: Any) -> str:
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, CompanionRunContext):
            parts = [p for p in (ctx.major, ctx.intent, ctx.handler) if p]
            if parts:
                return " / ".join(parts)
        return ""

    @staticmethod
    def _ends_with_tool_result(messages: list[Any]) -> bool:
        for msg in reversed(messages or []):
            if isinstance(msg, ToolMessage):
                return True
            if isinstance(msg, dict) and str(msg.get("role") or "") == "tool":
                return True
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if str(content or "").strip():
                return False
        return False

    def _sub_label(self, runtime: Any, messages: list[Any] | None = None) -> str:
        stage = self._stage()
        route = self._route_label(runtime)
        if stage == STAGE_AGENT:
            phase = "结合工具返回结果" if self._ends_with_tool_result(messages or []) else "决策怎么用工具"
            decide_n = 0
            for item in take_llm_trace():
                if str(item.get("stage") or "") != stage:
                    continue
                lab = str(item.get("label") or "")
                if route and route not in lab:
                    continue
                if "决策怎么用工具" in lab:
                    decide_n += 1
            group = decide_n + 1 if phase == "决策怎么用工具" else max(decide_n, 1)
            sub = f"第 {group} 组 · {phase}"
        else:
            same = 0
            for item in take_llm_trace():
                if str(item.get("stage") or "") != stage:
                    continue
                lab = str(item.get("label") or "")
                if route:
                    if route in lab:
                        same += 1
                else:
                    same += 1
            sub = f"第 {same + 1} 次 model 调用"
        if route:
            sub = f"{sub} · {route}"
        return sub

    @staticmethod
    def _messages_from_request(request: ModelRequest[Any]) -> list[Any]:
        msgs: list[Any] = []
        if request.system_message is not None:
            msgs.append(request.system_message)
        msgs.extend(list(request.messages or []))
        return msgs

    @staticmethod
    def _tool_calls_from_response(response: ModelResponse[Any] | AIMessage) -> list[Any]:
        if isinstance(response, AIMessage):
            return list(getattr(response, "tool_calls", None) or [])
        if isinstance(response, ModelResponse):
            calls: list[Any] = []
            for msg in list(response.result or []):
                if isinstance(msg, AIMessage):
                    calls.extend(list(getattr(msg, "tool_calls", None) or []))
            return calls
        return []

    @staticmethod
    def _text_response_only(response: ModelResponse[Any] | AIMessage) -> str:
        if isinstance(response, AIMessage):
            return str(response.content or "").strip()
        if isinstance(response, ModelResponse):
            parts: list[str] = []
            for msg in list(response.result or []):
                if isinstance(msg, AIMessage):
                    parts.append(str(msg.content or ""))
                else:
                    parts.append(str(msg))
            if response.structured_response is not None:
                try:
                    parts.append(str(response.structured_response))
                except Exception:
                    pass
            return "\n".join(p for p in parts if p).strip()
        return ""

    def _record_request(self, request: ModelRequest[Any], *, response: ModelResponse[Any] | AIMessage) -> None:
        route = self._route_label(request.runtime)
        tool_calls = self._tool_calls_from_response(response)
        msgs = self._messages_from_request(request)
        record_llm_call(
            self._stage(),
            serialize_langchain_messages(msgs),
            response=self._text_response_only(response),
            label=self._sub_label(request.runtime, msgs),
        )
        if isinstance(response, AIMessage):
            note_token_usage(response)
        elif isinstance(response, ModelResponse):
            for msg in list(response.result or []):
                if isinstance(msg, AIMessage):
                    note_token_usage(msg)
        if tool_calls:
            from algorithm.stream_debug import emit_trace_tool_calls

            emit_trace_tool_calls(tool_calls, route=route)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        response = handler(request)
        self._record_request(request, response=response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        response = await handler(request)
        self._record_request(request, response=response)
        return response

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        del state, runtime
        return None

    async def abefore_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:
        del state, runtime
        return None
