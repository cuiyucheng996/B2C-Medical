"""AgentLoop：依赖注入 + 最多 20 步 + 并行 Tool + emit/yield 双通道。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from src.langchain.tools.registry import ToolRegistry, ensure_tools_registered
from src.llm.tool_calls_accumulator import CompletedToolCall
from src.schemas.agent_protocol import SubAgentResult

from .backend import LLMBackend, OpenAICompatibleBackend
from .chunks import LLMChunk
from .context import LoopContext
from .events import AgentEvent, EventBus, EventType

logger = logging.getLogger(__name__)

MAX_STEPS = 20

__all__ = ["AgentLoop", "MAX_STEPS", "SessionStore"]


class SessionStore(Protocol):
    """子 Agent HITL 存档：钥匙是 session_id / resume_token，不是问诊图 thread_id。"""

    def save_snapshot(
        self,
        session_id: str,
        *,
        tool_call_id: str = "",
        agent_name: str = "",
        state_json: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> None: ...

    def get_pending_snapshot(self, session_id: str | None) -> dict[str, Any] | None: ...

    def resume_snapshot(self, snapshot_id: str) -> None: ...


class MemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def save_snapshot(
        self,
        session_id: str,
        *,
        tool_call_id: str = "",
        agent_name: str = "",
        state_json: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> None:
        snapshot_id = f"{session_id}:{tool_call_id or 'pending'}"
        self._data[session_id] = {
            "snapshot_id": snapshot_id,
            "pending": True,
            "tool_call_id": tool_call_id,
            "agent_name": agent_name,
            "state_json": dict(state_json or {}),
            "question": question,
        }

    def get_pending_snapshot(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        row = self._data.get(session_id)
        if not row or not row.get("pending"):
            return None
        return dict(row)

    def resume_snapshot(self, snapshot_id: str) -> None:
        for row in self._data.values():
            if row.get("snapshot_id") == snapshot_id:
                row["pending"] = False
                return


class AgentLoop:
    """主循环。LLM / 注册表 / 总线 / 上下文均由外部注入，便于 B/C 端复用与单测替换。"""

    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        registry: ToolRegistry | None = None,
        event_bus: EventBus | None = None,
        context: LoopContext | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        *,
        langchain_tools: list[BaseTool] | None = None,
    ) -> None:
        self.llm = llm_backend or OpenAICompatibleBackend()
        self.registry = registry or ensure_tools_registered()
        self.event_bus = event_bus or EventBus()
        self.context = context or LoopContext()
        self.session_store = session_store or MemorySessionStore()
        self.session_id = session_id
        self._langchain_tools = langchain_tools

    def _create_event(self, event_type: EventType, data: dict[str, Any] | None = None) -> AgentEvent:
        return AgentEvent(type=event_type, data=data or {})

    def _publish(self, event: AgentEvent) -> AgentEvent:
        self.event_bus.emit(event)
        return event

    def _tools(self) -> list[BaseTool]:
        if self._langchain_tools is not None:
            return self._langchain_tools
        return self.registry.langchain_tools_all()

    def _dispatch_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        emit_fn: Callable[[dict[str, Any]], None],
    ) -> SubAgentResult:
        """注入的 LangChain Tool 按名 invoke；否则走 Registry。"""
        for tool in self._tools():
            if getattr(tool, "name", None) == name:
                payload = {k: v for k, v in arguments.items() if k != "_session_id"}
                raw = tool.invoke(payload)
                if isinstance(raw, SubAgentResult):
                    return raw
                return SubAgentResult.done(str(raw))
        exec_args = {**arguments, "_session_id": self.session_id}
        return self.registry.execute(name, exec_args, emit_fn)

    async def run(self, user_message: str) -> AsyncIterator[AgentEvent]:
        self.context.add_user_message(user_message)
        async for event in self.iterate():
            yield event

    async def iterate(self) -> AsyncIterator[AgentEvent]:
        yield self._publish(self._create_event(EventType.AGENT_START, {"agent": "main"}))
        steps = 0
        while steps < MAX_STEPS:
            steps += 1
            if self.context.needs_compression():
                self.context.compress()

            messages = self.context.get_messages()
            tools = self._tools()
            text_buf: list[str] = []
            think_buf: list[str] = []
            done_chunk: LLMChunk | None = None

            try:
                async for chunk in self.llm.stream(
                    system=self.context.system_prompt,
                    messages=messages,
                    tools=tools or None,
                ):
                    if chunk.kind == "text":
                        text_buf.append(chunk.text)
                        yield self._publish(
                            self._create_event(EventType.TEXT_CHUNK, {"content": chunk.text})
                        )
                    elif chunk.kind == "think":
                        think_buf.append(chunk.text)
                        yield self._publish(
                            self._create_event(EventType.THINK_CHUNK, {"content": chunk.text})
                        )
                    elif chunk.kind == "tool_calls_delta" and chunk.tool_calls_delta is not None:
                        delta = chunk.tool_calls_delta
                        yield self._publish(
                            self._create_event(
                                EventType.TOOL_CALLS_DELTA,
                                {
                                    "index": delta.index,
                                    "id": delta.id,
                                    "name": delta.function_name,
                                    "function_arguments": delta.function_arguments or "",
                                },
                            )
                        )
                    elif chunk.kind == "done":
                        done_chunk = chunk
                        break
            except asyncio.CancelledError:
                self.context.heal_dangling_tool_calls("任务已取消")
                raise
            except Exception as exc:
                logger.exception("AgentLoop LLM stream failed")
                async for event in self._graceful_fail(f"LLM 调用异常: {exc}"):
                    yield event
                return

            calls = list(done_chunk.tool_calls or []) if done_chunk else []
            full_text = (done_chunk.full_text if done_chunk else "") or "".join(text_buf)
            full_think = (done_chunk.full_think if done_chunk else "") or "".join(think_buf)

            if not calls:
                self.context.add_ai_message(full_text, think=full_think)
                yield self._publish(
                    self._create_event(
                        EventType.AGENT_END,
                        {"agent": "main", "text": full_text, "think": full_think},
                    )
                )
                yield self._publish(
                    self._create_event(
                        EventType.DONE,
                        {"text": full_text, "think": full_think},
                    )
                )
                return

            lc_calls = [
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.parsed_arguments(),
                    "type": "tool_call",
                }
                for call in calls
            ]
            self.context.add_ai_message(full_text, tool_calls=lc_calls, think=full_think)

            try:
                results = await self._execute_tools(calls)
            except asyncio.CancelledError:
                self.context.heal_dangling_tool_calls("任务已取消")
                raise
            for call, outcome in zip(calls, results, strict=True):
                result: SubAgentResult = outcome
                tool_end_data: dict[str, Any] = {
                    "tool_call_id": call.id,
                    "name": call.name,
                    "status": result.status,
                    "output": result.output,
                    "ok": result.status != "failed",
                }
                if result.snapshot and result.status == "done":
                    tool_end_data["meta"] = result.snapshot
                yield self._publish(self._create_event(EventType.TOOL_END, tool_end_data))

                if result.status == "need_input":
                    question = result.question or result.output or "[等待用户确认]"
                    self.context.add_tool_result(
                        tool_call_id=call.id,
                        name=call.name,
                        content=question,
                    )
                    self.context.heal_dangling_tool_calls(
                        "工具未执行：本轮已挂起等待用户输入"
                    )
                    yield self._publish(
                        self._create_event(EventType.TEXT_CHUNK, {"content": question})
                    )
                    if self.session_id:
                        # need_input：写入 SessionStore，前端用 resume_token 续跑子 Agent
                        self.session_store.save_snapshot(
                            self.session_id,
                            tool_call_id=call.id,
                            agent_name=call.name,
                            state_json=result.snapshot or {},
                            question=result.question,
                        )
                    yield self._publish(
                        self._create_event(
                            EventType.SUSPENDED,
                            {
                                "tool_call_id": call.id,
                                "agent_name": call.name,
                                "question": result.question,
                            },
                        )
                    )
                    yield self._publish(self._create_event(EventType.DONE, {}))
                    return

                output = result.output or result.error or ""
                self.context.add_tool_result(
                    tool_call_id=call.id,
                    name=call.name,
                    content=output,
                    artifacts=list(result.artifacts or []),
                )
                for art in result.artifacts or []:
                    payload = art.to_dict() if hasattr(art, "to_dict") else dict(art)
                    yield self._publish(self._create_event(EventType.ARTIFACT, payload))
        else:
            async for event in self._graceful_fail(f"超过最大循环轮数 {MAX_STEPS}"):
                yield event

    async def _graceful_fail(self, reason: str) -> AsyncIterator[AgentEvent]:
        """配对修干净 → error → done，不把异常抛出打成 500。"""
        self.context.heal_dangling_tool_calls(reason)
        err_msg = f"抱歉，模型服务调用失败（{reason}），请稍后重试。"
        yield self._publish(self._create_event(EventType.ERROR, {"error": err_msg, "message": err_msg}))
        yield self._publish(self._create_event(EventType.DONE, {}))

    async def _execute_tools(self, tool_calls: list[CompletedToolCall]) -> list[SubAgentResult]:
        async def execute_one(call: CompletedToolCall) -> SubAgentResult:
            emit_fn = self.event_bus.create_emit_fn(source=f"subagent:{call.name}")
            try:
                arguments = dict(call.parsed_arguments())
                if arguments.get("resume_token") and self.session_id:
                    snapshot = self.session_store.get_pending_snapshot(self.session_id)
                    if snapshot:
                        state = snapshot.get("state_json") or {}
                        for key, value in state.items():
                            if key not in arguments:
                                arguments[key] = value
                        snapshot_id = str(snapshot.get("snapshot_id") or "")
                        if snapshot_id:
                            self.session_store.resume_snapshot(snapshot_id)

                self._publish(
                    self._create_event(
                        EventType.TOOL_START,
                        {
                            "tool_call_id": call.id,
                            "name": call.name,
                            "arguments": arguments,
                            "source": f"subagent:{call.name}",
                        },
                    )
                )
                return await asyncio.to_thread(
                    self._dispatch_tool, call.name, arguments, emit_fn
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("tool %s failed", call.name)
                return SubAgentResult.failed(f"工具 {call.name} 执行异常: {exc}")

        raw_results = await asyncio.gather(
            *[execute_one(call) for call in tool_calls],
            return_exceptions=True,
        )
        results: list[SubAgentResult] = []
        for raw in raw_results:
            if isinstance(raw, SubAgentResult):
                results.append(raw)
            elif isinstance(raw, asyncio.CancelledError):
                raise raw
            else:
                results.append(SubAgentResult.failed(f"工具执行异常: {raw}"))
        return results


def run_loop_sync(
    context: LoopContext,
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
    registry: ToolRegistry | None = None,
    session_id: str | None = None,
    langchain_tools: list[BaseTool] | None = None,
    llm_backend: LLMBackend | None = None,
) -> list[BaseMessage]:
    """给 LangGraph 同步节点用：跑完 AgentLoop，返回含 ToolMessage 的 transcript。"""
    bus = EventBus()
    if emit is not None:
        bus.subscribe(lambda event: emit(event.to_sse()))
    loop_agent = AgentLoop(
        llm_backend=llm_backend,
        registry=registry or ensure_tools_registered(),
        event_bus=bus,
        context=context,
        session_id=session_id,
        langchain_tools=langchain_tools,
    )

    async def _consume() -> list[BaseMessage]:
        async for _event in loop_agent.iterate():
            pass
        return loop_agent.context.get_messages()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_consume())

    result: list[BaseMessage] = []

    def _thread() -> None:
        nonlocal result
        result = asyncio.run(_consume())

    import threading

    worker = threading.Thread(target=_thread, name="agent-loop", daemon=True)
    worker.start()
    worker.join(timeout=600)
    return result
