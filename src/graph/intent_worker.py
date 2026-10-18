"""单意图子图：执行 Agent → 评审 Agent；不通过则限轮回环。"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable, Literal

from langgraph.graph import END, START, StateGraph

from graph.review import companion_should_review_draft, review_companion_draft
from graph.state import IntentWorkerState
from middleware.middleware_trace import take_middleware_trace
from middleware.tool_trace import take_tool_trace
from tools.companion_tools import tools_for_intent
from tools.mcp_bridge import mcp_tools_for_major

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_REVIEW_RETRIES",
    "build_intent_worker_graph",
    "run_intent_worker",
    "runtime_model",
    "runtime_merge_on_token",
    "runtime_on_token",
    "runtime_stream_lock",
    "runtime_stream_sep",
    "runtime_worker_done_token",
]

MAX_REVIEW_RETRIES = 1
runtime_model: ContextVar[Any] = ContextVar("companion_graph_model", default=None)
runtime_on_token: ContextVar[Any] = ContextVar("companion_graph_on_token", default=None)
runtime_merge_on_token: ContextVar[Any] = ContextVar("companion_graph_merge_on_token", default=None)
runtime_worker_done_token: ContextVar[Any] = ContextVar("companion_worker_done_token", default=None)
runtime_stream_sep: ContextVar[dict[str, bool] | None] = ContextVar("companion_stream_sep", default=None)
runtime_stream_lock: ContextVar[Any] = ContextVar("companion_stream_lock", default=None)
runtime_stream_gate: ContextVar[Any] = ContextVar("companion_stream_gate", default=None)
runtime_stream_worker_id: ContextVar[str] = ContextVar("companion_stream_worker_id", default="")


class ExclusiveTokenGate:
    """多路并行时同一时刻只把一路的模型 token 推到前端，其它路先缓冲。"""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.holder: str | None = None
        self.buffers: dict[str, list[str]] = {}
        self.done: set[str] = set()
        self.used = False
        self._lock = threading.Lock()

    def feed(self, token: str) -> None:
        text = str(token or "")
        if not text:
            return
        wid = runtime_stream_worker_id.get() or "default"
        with self._lock:
            self.used = True
            if self.holder is None or self.holder == wid:
                if self.holder is None and self.used and self.done:
                    self.emit("\n\n")
                self.holder = wid
                self.emit(text)
                return
            self.buffers.setdefault(wid, []).append(text)

    def release(self, wid: str) -> None:
        with self._lock:
            self.done.add(wid)
            while True:
                if self.holder and self.holder not in self.done:
                    return
                nxt = next((key for key, chunks in self.buffers.items() if chunks), None)
                if nxt is None:
                    self.holder = None
                    return
                chunks = self.buffers.pop(nxt, [])
                self.emit("\n\n")
                for piece in chunks:
                    self.emit(piece)
                self.holder = nxt
                if nxt not in self.done:
                    return
                self.holder = None


def _task_row(state: IntentWorkerState) -> dict[str, Any]:
    row = state.get("task") if isinstance(state.get("task"), dict) else {}
    return dict(row)


async def execute_node(state: IntentWorkerState) -> dict[str, Any]:
    from agent.companion_agent import _ainvoke_companion_agent

    model = runtime_model.get()
    task = _task_row(state)
    observe = str(task.get("observe") or state.get("observe") or "")
    peer = str(state.get("peer_brief") or "").strip()
    if peer:
        observe = f"{observe}\n\n【同伴子智能体 handoff 摘要】\n{peer}"
    hint = str(state.get("review_hint") or "").strip()
    if hint:
        observe = f"{observe}\n\n【评审打回】{hint}"
    if str(task.get("major") or "") == "公共知识问答" and int(state.get("review_round") or 0) == 0:
        from algorithm.kg_query_plan import format_query_plan_observe, plan_knowledge_queries

        plan = plan_knowledge_queries(model, str(state.get("query") or ""))
        observe = f"{observe}\n\n{format_query_plan_observe(plan)}"
    if str(task.get("handler") or "") == "report_sync":
        from algorithm.lab_report_entities import extract_lab_sync_entities, format_lab_sync_observe

        lab_entities = extract_lab_sync_entities(str(state.get("query") or ""))
        observe = f"{observe}\n\n{format_lab_sync_observe(lab_entities)}"
    else:
        lab_entities = {}
    turn = {
        **dict(state.get("base_turn") or {}),
        "handler": task.get("handler") or (state.get("base_turn") or {}).get("handler") or "",
        "lab_sync_entities": lab_entities,
    }
    handler = str(task.get("handler") or "")
    local = list(tools_for_intent(major=str(task.get("major") or ""), handler=handler))
    if handler == "mem_session_recall":
        extra = []
    else:
        extra = await mcp_tools_for_major(str(task.get("major") or ""))
    have = {str(getattr(tool, "name", "") or "") for tool in local}
    tools = local + [tool for tool in extra if str(getattr(tool, "name", "") or "") not in have]
    bound_names = [str(getattr(tool, "name", "") or "") for tool in tools if getattr(tool, "name", None)]
    wid = "|".join(
        [
            str(task.get("major") or ""),
            str(task.get("intent") or ""),
            str(state.get("review_round") or 0),
        ]
    )
    id_tok = runtime_stream_worker_id.set(wid)
    from algorithm.memory_collect import history_for_intent_worker

    base_turn = dict(state.get("base_turn") or {})
    worker_history = history_for_intent_worker(
        list(state.get("history") or []),
        handler=handler,
        major=str(task.get("major") or ""),
        user_id=str(base_turn.get("user_id") or ""),
    )
    try:
        reply, facts, tool_trace = await _ainvoke_companion_agent(
            model,
            tools,
            worker_history,
            observe,
            turn,
            task,
            on_token=runtime_on_token.get(),
        )
    finally:
        gate = runtime_stream_gate.get()
        if gate is not None:
            gate.release(wid)
        runtime_stream_worker_id.reset(id_tok)
    prior_mw = list(state.get("middleware_trace") or [])
    prior_tools = list(state.get("tool_trace") or [])
    merged_tools = prior_tools + list(tool_trace or take_tool_trace())
    return {
        "draft": reply,
        "facts": list(facts or []),
        "tool_trace": merged_tools,
        "middleware_trace": prior_mw + take_middleware_trace(),
        "bound_tools": bound_names,
    }


async def review_node(state: IntentWorkerState) -> dict[str, Any]:
    import asyncio

    from algorithm.llm_trace import run_in_llm_trace_context

    task = _task_row(state)
    major = str(task.get("major") or "")
    handler = str(task.get("handler") or "")
    if not companion_should_review_draft(major, handler):
        return {
            "review_pass": True,
            "review_reason": "",
            "review_hint": "",
        }

    def _run() -> dict[str, Any]:
        return review_companion_draft(
            query=str(state.get("query") or ""),
            major=str(task.get("major") or ""),
            intent=str(task.get("intent") or ""),
            handler=str(task.get("handler") or ""),
            draft=str(state.get("draft") or ""),
        )

    verdict = await asyncio.to_thread(run_in_llm_trace_context, _run)
    passed = bool(verdict.get("pass"))
    rnd = int(state.get("review_round") or 0)
    if passed:
        return {
            "review_pass": True,
            "review_reason": str(verdict.get("reason") or ""),
            "review_hint": "",
        }
    return {
        "review_pass": False,
        "review_reason": str(verdict.get("reason") or ""),
        "review_hint": str(verdict.get("hint") or ""),
        "review_round": rnd + 1,
    }


def route_after_review(state: IntentWorkerState) -> Literal["execute", "finish"]:
    if state.get("review_pass"):
        return "finish"
    if int(state.get("review_round") or 0) > MAX_REVIEW_RETRIES:
        return "finish"
    return "execute"


def finish_node(state: IntentWorkerState) -> dict[str, Any]:
    task = _task_row(state)
    return {
        "facts": list(state.get("facts") or []),
        "worker_outputs": [
            {
                "reply": str(state.get("draft") or "").strip(),
                "major": str(task.get("major") or ""),
                "intent": str(task.get("intent") or ""),
                "handler": str(task.get("handler") or ""),
                "review_pass": bool(state.get("review_pass")),
                "review_reason": str(state.get("review_reason") or ""),
                "review_retries": int(state.get("review_round") or 0),
                "tool_trace": list(state.get("tool_trace") or []),
                "middleware_trace": list(state.get("middleware_trace") or []),
                "bound_tools": list(state.get("bound_tools") or []),
            }
        ],
    }


@lru_cache(maxsize=1)
def build_intent_worker_graph() -> Any:
    builder = StateGraph(IntentWorkerState)
    builder.add_node("execute", execute_node)
    builder.add_node("review", review_node)
    builder.add_node("finish", finish_node)
    builder.add_edge(START, "execute")
    builder.add_edge("execute", "review")
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"execute": "execute", "finish": "finish"},
    )
    builder.add_edge("finish", END)
    return builder.compile()


async def _emit_finished_worker(outputs: list[dict[str, Any]]) -> None:
    """子图整段结束后立刻推 SSE，不等其它并行路。"""
    import asyncio

    from graph.companion import _format_worker_stream_block, emit_reply_chunks

    gate = runtime_stream_gate.get()
    if gate is not None and getattr(gate, "used", False):
        return
    on_token = runtime_worker_done_token.get()
    if not on_token:
        return
    lock = runtime_stream_lock.get()
    sep = runtime_stream_sep.get()
    if lock is None or sep is None:
        return

    async with lock:
        for item in outputs:
            if not isinstance(item, dict):
                continue
            block = _format_worker_stream_block(item, multi=True)
            if not block:
                continue
            if sep.get("sep"):
                on_token("\n\n")
            else:
                sep["sep"] = True
            emit_reply_chunks(block, on_token)
    await asyncio.sleep(0)


async def run_intent_worker(state: IntentWorkerState) -> dict[str, Any]:
    """子图跑完只把可合并字段写回主图，避免 query 等 LastValue 并发冲突。"""
    result = await build_intent_worker_graph().ainvoke(state)
    if not isinstance(result, dict):
        return {"worker_outputs": [], "facts": []}
    outputs = list(result.get("worker_outputs") or [])
    await _emit_finished_worker(outputs)
    return {
        "worker_outputs": outputs,
        "facts": list(result.get("facts") or []),
    }
