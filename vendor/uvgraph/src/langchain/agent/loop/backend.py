"""OpenAI 兼容 Backend：同步 LangChain stream → 线程 + 队列 → 异步 LLMChunk。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from src.langchain.agent.stream_agent import chunk_text
from src.llm.chat import get_chat_model
from src.llm.tool_calls_accumulator import ToolCallsAccumulator

from .chunks import LLMChunk, ToolCallDelta

logger = logging.getLogger(__name__)

__all__ = ["LLMBackend", "OpenAICompatibleBackend"]


@runtime_checkable
class LLMBackend(Protocol):
    def stream(
        self,
        system: str,
        messages: list[BaseMessage],
        tools: list[BaseTool] | None = None,
    ) -> AsyncIterator[LLMChunk]: ...


def _reasoning_from_chunk(chunk: Any) -> str:
    extra = getattr(chunk, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = extra.get(key)
        if value:
            return str(value)
    meta = getattr(chunk, "response_metadata", None) or {}
    value = meta.get("reasoning_content") or meta.get("reasoning")
    return str(value) if value else ""


class OpenAICompatibleBackend:
    """同步 `model.stream` 不阻塞 asyncio 事件循环。"""

    def __init__(
        self,
        llm_client_factory: Callable[[], Any] | None = None,
        *,
        enable_think_filter: bool = True,
    ) -> None:
        self._factory = llm_client_factory or get_chat_model
        self._enable_think_filter = enable_think_filter

    async def stream(
        self,
        system: str,
        messages: list[BaseMessage],
        tools: list[BaseTool] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        model = self._factory()
        if model is None:
            raise RuntimeError("LLM 未配置")

        bound = model.bind_tools(tools) if tools else model
        transcript = list(messages)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        def _run_sync() -> None:
            try:
                for chunk in bound.stream(transcript):
                    fut = asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop)
                    fut.result(timeout=600)
            except Exception as exc:
                fut = asyncio.run_coroutine_threadsafe(queue.put(("_error", str(exc))), loop)
                fut.result(timeout=30)
            finally:
                fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                fut.result(timeout=30)

        loop.run_in_executor(None, _run_sync)

        text_buf: list[str] = []
        think_buf: list[str] = []
        accumulator = ToolCallsAccumulator()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, tuple) and item and item[0] == "_error":
                raise RuntimeError(f"LLM 服务异常: {item[1]}")
            _, chunk = item
            text = chunk_text(getattr(chunk, "content", None))
            if text:
                text_buf.append(text)
                yield LLMChunk(kind="text", text=text)
            reasoning = _reasoning_from_chunk(chunk)
            if reasoning and self._enable_think_filter:
                think_buf.append(reasoning)
                yield LLMChunk(kind="think", text=reasoning)
            for tc in getattr(chunk, "tool_call_chunks", None) or []:
                if isinstance(tc, dict):
                    delta = ToolCallDelta(
                        index=int(tc.get("index", 0)),
                        id=tc.get("id"),
                        function_name=tc.get("name"),
                        function_arguments=str(tc.get("args") or "") or None,
                    )
                else:
                    delta = ToolCallDelta(
                        index=int(getattr(tc, "index", 0)),
                        id=getattr(tc, "id", None),
                        function_name=getattr(tc, "name", None),
                        function_arguments=str(getattr(tc, "args", "") or "") or None,
                    )
                accumulator.feed_delta(delta)
                yield LLMChunk(kind="tool_calls_delta", tool_calls_delta=delta)

        yield LLMChunk(
            kind="done",
            full_text="".join(text_buf),
            full_think="".join(think_buf),
            tool_calls=accumulator.flush(),
        )
