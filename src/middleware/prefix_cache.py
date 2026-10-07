"""子 Agent 多轮 model 调用：稳定 system 块打 cache_control。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from algorithm.prefix_cache import (
    cache_mark_message,
    companion_prefix_cache_enabled,
)

from middleware.context import CompanionRunContext

__all__ = ["CompanionPrefixCacheMiddleware"]


def _system_text(sm: SystemMessage | None) -> str:
    if sm is None:
        return ""
    content = getattr(sm, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _already_cached(sm: SystemMessage) -> bool:
    content = getattr(sm, "content", None)
    if not isinstance(content, list) or not content:
        return False
    for block in content:
        if isinstance(block, dict) and block.get("cache_control"):
            return True
    return False


class CompanionPrefixCacheMiddleware(AgentMiddleware):
    def _with_prefix_cache(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        if not companion_prefix_cache_enabled():
            return request
        req = request
        sm = req.system_message
        if sm is not None and not _already_cached(sm):
            text = _system_text(sm)
            if text.strip():
                from algorithm.prefix_cache import cached_system_message

                req = req.override(system_message=cached_system_message(text))
        ctx = getattr(req.runtime, "context", None)
        if not isinstance(ctx, CompanionRunContext):
            return req
        msgs = list(req.messages or [])
        if not msgs:
            return req
        if ctx.prefix_message_count <= 0:
            ctx.prefix_message_count = len(msgs)
        n = min(ctx.prefix_message_count, len(msgs))
        marked = [cache_mark_message(m) if i < n else m for i, m in enumerate(msgs)]
        if marked != msgs:
            req = req.override(messages=marked)
        return req

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._with_prefix_cache(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._with_prefix_cache(request))
