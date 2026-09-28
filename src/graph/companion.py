"""陪聊主图：Send 并行意图子图 → merge 合并（可选 peer_handoff 二轮，默认关）。"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.begin_turn import begin_turn_node
from graph.checkpointer import (
    companion_checkpoint_enabled,
    companion_graph_thread_id,
    companion_invoke_config,
    get_companion_checkpointer,
)
from graph.intent_worker import (
    ExclusiveTokenGate,
    run_intent_worker,
    runtime_merge_on_token,
    runtime_model,
    runtime_on_token,
    runtime_stream_gate,
    runtime_stream_lock,
    runtime_stream_sep,
    runtime_worker_done_token,
)
from graph.peer_handoff import (
    dispatch_handoff_intents,
    peer_handoff_node,
    route_after_intent_worker,
)
from graph.checkpoint_lifecycle import maybe_release_after_run
from graph.state import CompanionGraphState

__all__ = [
    "ainvoke_companion_intent_graph",
    "ainvoke_send_companion_graph",
    "build_companion_graph",
    "emit_merge_reply_stream",
    "emit_reply_chunks",
    "merge_node",
]

_MERGE_STREAM_CHUNK = 18


def emit_reply_chunks(reply: str, on_token: Any) -> None:
    text = (reply or "").strip()
    if not text or not on_token or not callable(on_token):
        return
    for i in range(0, len(text), _MERGE_STREAM_CHUNK):
        on_token(text[i : i + _MERGE_STREAM_CHUNK])


def emit_merge_reply_stream(reply: str, on_token: Any, *, worker_count: int) -> None:
    """多 worker 合并后的整段回复分块推前端（单 worker 仍走子图 CompanionReply 流式）。"""
    if worker_count <= 1:
        return
    emit_reply_chunks(reply, on_token)


def _task_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "major": str(task.get("major") or ""),
        "intent": str(task.get("intent") or ""),
        "handler": str(task.get("handler") or ""),
        "observe": str(task.get("observe") or ""),
    }


def _tasks_for_graph(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """主图 state / checkpoint 只保留可序列化字段（tools 在 worker 内按 handler 再绑）。"""
    rows: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        rows.append(_task_payload(item))
    return rows


def _worker_output_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("major") or ""),
        str(item.get("intent") or ""),
        str(item.get("handler") or ""),
    )


def _merge_worker_outputs_for_trace(
    wave1: list[dict[str, Any]],
    wave2: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """同伴 handoff 二轮会清空 worker_outputs；工具 trace 多在 wave1，需与 wave2 按任务合并。"""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []

    def absorb(item: dict[str, Any], *, prefer_reply: bool) -> None:
        key = _worker_output_key(item)
        if key not in merged:
            merged[key] = {
                "major": item.get("major"),
                "intent": item.get("intent"),
                "handler": item.get("handler"),
                "tool_trace": list(item.get("tool_trace") or []),
                "middleware_trace": list(item.get("middleware_trace") or []),
                "bound_tools": list(item.get("bound_tools") or []),
                "review_pass": item.get("review_pass"),
                "review_reason": item.get("review_reason"),
                "review_retries": item.get("review_retries"),
                "reply": str(item.get("reply") or ""),
            }
            order.append(key)
            return
        row = merged[key]
        row["tool_trace"] = list(row.get("tool_trace") or []) + list(item.get("tool_trace") or [])
        row["middleware_trace"] = list(row.get("middleware_trace") or []) + list(
            item.get("middleware_trace") or []
        )
        bound = list(row.get("bound_tools") or [])
        for name in list(item.get("bound_tools") or []):
            n = str(name or "").strip()
            if n and n not in bound:
                bound.append(n)
        row["bound_tools"] = bound
        if prefer_reply and str(item.get("reply") or "").strip():
            row["reply"] = str(item.get("reply") or "")
            row["review_pass"] = item.get("review_pass")
            row["review_reason"] = item.get("review_reason")
            row["review_retries"] = item.get("review_retries")

    for item in wave1:
        if isinstance(item, dict):
            absorb(item, prefer_reply=False)
    for item in wave2:
        if isinstance(item, dict):
            absorb(item, prefer_reply=True)
    return [merged[k] for k in order]


def _format_worker_stream_block(item: dict[str, Any], *, multi: bool) -> str:
    body = str(item.get("reply") or "").strip()
    if not body:
        return ""
    if not multi:
        return body
    label = " / ".join(
        p
        for p in (
            str(item.get("major") or "").strip(),
            str(item.get("intent") or "").strip(),
        )
        if p
    )
    return f"【{label}】\n{body}" if label else body


def dispatch_intents(state: CompanionGraphState) -> list[Send]:
    tasks = [item for item in list(state.get("tasks") or []) if isinstance(item, dict)]
    if not tasks:
        return [Send("merge", state)]
    sends: list[Send] = []
    for task in tasks:
        payload = _task_payload(task)
        sends.append(
            Send(
                "intent_worker",
                {
                    "query": str(state.get("query") or ""),
                    "observe": payload["observe"] or str(state.get("observe") or ""),
                    "history": list(state.get("history") or []),
                    "base_turn": dict(state.get("base_turn") or {}),
                    "task": payload,
                    "review_hint": "",
                    "review_round": 0,
                    "draft": "",
                    "facts": [],
                    "review_pass": False,
                    "review_reason": "",
                    "worker_outputs": [],
                },
            )
        )
    return sends


def merge_node(state: CompanionGraphState) -> dict[str, Any]:
    rows = [item for item in list(state.get("worker_outputs") or []) if isinstance(item, dict)]
    from algorithm.merge_reply import merge_worker_replies

    text = merge_worker_replies(rows)
    wave1 = [item for item in list(state.get("wave1_outputs") or []) if isinstance(item, dict)]
    trace_rows = _merge_worker_outputs_for_trace(wave1, rows) if wave1 else rows
    pattern = "send+peer_handoff+review" if wave1 else "send+subgraph+review"
    base_turn = state.get("base_turn") if isinstance(state.get("base_turn"), dict) else {}
    trace_id = str(base_turn.get("trace_id") or state.get("trace_id") or "").strip()
    from graph.checkpointer import companion_checkpoint_enabled
    from graph.checkpoint_lifecycle import checkpoint_retention_mode

    final = text or "我在，你可以继续说说近况。"
    from graph.intent_worker import runtime_worker_done_token

    if runtime_worker_done_token.get() is None:
        emit_merge_reply_stream(
            final,
            runtime_merge_on_token.get(),
            worker_count=len(rows),
        )

    return {
        "reply": final,
        "trace": {
            "engine": "langgraph",
            "pattern": pattern,
            "workers": len(rows),
            "peer_handoff": bool(wave1),
            "wave1_workers": len(wave1),
            "trace_id": trace_id,
            "thread_id": str(state.get("thread_id") or ""),
            "turn_count": int(state.get("turn_count") or 0),
            "checkpointer": companion_checkpoint_enabled(),
            "checkpoint_retention": checkpoint_retention_mode(),
            "reviews": [
                {
                    "major": item.get("major"),
                    "intent": item.get("intent"),
                    "pass": item.get("review_pass"),
                    "reason": item.get("review_reason"),
                    "retries": item.get("review_retries"),
                }
                for item in trace_rows
            ],
            "tool_runs": [
                {
                    "major": item.get("major"),
                    "intent": item.get("intent"),
                    "handler": item.get("handler"),
                    "bound_tools": list(item.get("bound_tools") or []),
                    "steps": list(item.get("tool_trace") or []),
                    "middleware_trace": list(item.get("middleware_trace") or []),
                }
                for item in trace_rows
            ],
        },
    }


@lru_cache(maxsize=1)
def build_companion_graph() -> Any:
    builder = StateGraph(CompanionGraphState)
    builder.add_node("begin_turn", begin_turn_node)
    builder.add_node("intent_worker", run_intent_worker)
    builder.add_node("peer_handoff", peer_handoff_node)
    builder.add_node("merge", merge_node)
    builder.add_edge(START, "begin_turn")
    builder.add_conditional_edges("begin_turn", dispatch_intents, ["intent_worker", "merge"])
    builder.add_conditional_edges(
        "intent_worker",
        route_after_intent_worker,
        {"peer_handoff": "peer_handoff", "merge": "merge"},
    )
    builder.add_conditional_edges(
        "peer_handoff",
        dispatch_handoff_intents,
        ["intent_worker", "merge"],
    )
    builder.add_edge("merge", END)
    checkpointer = get_companion_checkpointer()
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


async def ainvoke_send_companion_graph(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any = None,
    *,
    thread_id: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not tasks:
        return "我在，你可以继续说说近况。", [], {"engine": "langgraph", "workers": 0}
    tid = companion_graph_thread_id(base_turn, explicit=thread_id)
    invoke_config = companion_invoke_config(tid) if companion_checkpoint_enabled() else None
    multi_stream = len(tasks) > 1 and on_token is not None
    gate = ExclusiveTokenGate(on_token) if multi_stream else None
    tok_m = runtime_model.set(model)
    tok_t = runtime_on_token.set(gate.feed if gate is not None else on_token)
    tok_merge = runtime_merge_on_token.set(None if multi_stream else on_token)
    tok_done = runtime_worker_done_token.set(on_token if multi_stream else None)
    tok_sep = runtime_stream_sep.set({"sep": False} if multi_stream else None)
    tok_lock = runtime_stream_lock.set(asyncio.Lock() if multi_stream else None)
    tok_gate = runtime_stream_gate.set(gate)
    result: Any = None
    companion_ckpt_snap: dict[str, Any] | None = None
    try:
        payload = {
            "thread_id": tid,
            "query": str((base_turn or {}).get("query") or ""),
            "observe": str((tasks[0] or {}).get("observe") or ""),
            "history": list(history or []),
            "base_turn": dict(base_turn or {}),
                "tasks": _tasks_for_graph(tasks),
            "reply": "",
            "trace": {},
        }
        if invoke_config is not None:
            result = await build_companion_graph().ainvoke(payload, config=invoke_config)
        else:
            result = await build_companion_graph().ainvoke(payload)
        from graph.checkpoint_debug import snapshot_companion_checkpoint_debug

        companion_ckpt_snap = snapshot_companion_checkpoint_debug(tid)
    finally:
        runtime_model.reset(tok_m)
        runtime_on_token.reset(tok_t)
        runtime_merge_on_token.reset(tok_merge)
        runtime_worker_done_token.reset(tok_done)
        runtime_stream_sep.reset(tok_sep)
        runtime_stream_lock.reset(tok_lock)
        runtime_stream_gate.reset(tok_gate)
        await maybe_release_after_run(tid)
    if not isinstance(result, dict):
        return "我在，你可以继续说说近况。", [], {}
    reply = str(result.get("reply") or "").strip() or "我在，你可以继续说说近况。"
    trace = dict(result.get("trace") or {})
    if companion_ckpt_snap is not None:
        trace["companion_checkpoint_snapshot"] = companion_ckpt_snap
    return (
        reply,
        list(result.get("facts") or []),
        trace,
    )


async def ainvoke_companion_intent_graph(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any = None,
    *,
    thread_id: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not tasks:
        return "我在，你可以继续说说近况。", [], {"engine": "langgraph", "workers": 0}
    from graph.supervisor_route import ainvoke_supervisor_companion_graph, companion_use_supervisor

    if companion_use_supervisor():
        return await ainvoke_supervisor_companion_graph(
            model, tasks, history, base_turn, on_token=on_token, thread_id=thread_id
        )
    return await ainvoke_send_companion_graph(
        model, tasks, history, base_turn, on_token=on_token, thread_id=thread_id
    )
