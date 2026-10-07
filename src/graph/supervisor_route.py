"""create_supervisor 主管 + transfer_to 子智能体；与 Send 图可切换。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

from graph.checkpoint_lifecycle import maybe_release_after_run
from graph.checkpointer import (
    companion_checkpoint_enabled,
    companion_graph_thread_id,
    companion_invoke_config,
    get_companion_checkpointer,
)
from graph.companion import emit_merge_reply_stream, merge_node
from graph.intent_worker import runtime_model, runtime_on_token
from graph.supervisor_state import CompanionSupervisorState, major_to_agent_name
from graph.supervisor_workers import major_worker_graphs_for_tasks
from prompt import load_prompt

logger = logging.getLogger(__name__)

__all__ = ["companion_use_supervisor", "ainvoke_supervisor_companion_graph"]


def companion_use_supervisor() -> bool:
    raw = (os.getenv("COMPANION_USE_SUPERVISOR") or "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _supervisor_user_message(ctx: dict[str, Any]) -> str:
    tasks = [
        {
            "major": t.get("major"),
            "intent": t.get("intent"),
            "handler": t.get("handler"),
            "transfer_tool": f"transfer_to_{major_to_agent_name(str(t.get('major') or 'other'))}",
        }
        for t in list(ctx.get("tasks") or [])
        if isinstance(t, dict)
    ]
    return (
        f"用户问题：{ctx.get('query') or ''}\n"
        f"预处理观察（节选）：{(ctx.get('observe') or '')[:1200]}\n"
        f"意图任务清单（JSON）：{json.dumps(tasks, ensure_ascii=False)}\n"
        "请按清单调用 transfer_to 交给子智能体。"
    )


def _build_supervisor_workflow(model: Any, tasks: list[dict[str, Any]]) -> Any:
    agents = major_worker_graphs_for_tasks(tasks)
    workflow = create_supervisor(
        agents,
        model=model,
        prompt=load_prompt("companion_supervisor"),
        state_schema=CompanionSupervisorState,
        parallel_tool_calls=True,
        output_mode="last_message",
        supervisor_name="companion_supervisor",
    )
    checkpointer = get_companion_checkpointer()
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


async def ainvoke_supervisor_companion_graph(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any = None,
    *,
    thread_id: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if model is None:
        return "我在，你可以继续说说近况。", [], {"engine": "supervisor", "error": "no_model"}
    tid = companion_graph_thread_id(base_turn, explicit=thread_id)
    invoke_config = companion_invoke_config(tid) if companion_checkpoint_enabled() else None
    ctx = {
        "query": str((base_turn or {}).get("query") or ""),
        "observe": str((tasks[0] or {}).get("observe") or ""),
        "history": list(history or []),
        "base_turn": dict(base_turn or {}),
        "tasks": list(tasks),
    }
    tok_m = runtime_model.set(model)
    tok_t = runtime_on_token.set(on_token if len(tasks) == 1 else None)
    result: Any = None
    companion_ckpt_snap: dict[str, Any] | None = None
    try:
        app = _build_supervisor_workflow(model, tasks)
        sup_payload = {
            "messages": [HumanMessage(content=_supervisor_user_message(ctx))],
            "companion_ctx": ctx,
            "worker_outputs": [],
            "facts": [],
            "reply": "",
            "trace": {"thread_id": tid},
        }
        from graph.checkpoint_debug import ainvoke_graph_emitting_checkpoints

        if invoke_config is not None:
            result = await ainvoke_graph_emitting_checkpoints(app, sup_payload, invoke_config)
        else:
            result = await app.ainvoke(sup_payload)
        from graph.checkpoint_debug import snapshot_companion_checkpoint_debug

        companion_ckpt_snap = snapshot_companion_checkpoint_debug(tid)
    except Exception:
        logger.exception("supervisor graph failed, fallback to send graph")
        return await _fallback_send(model, tasks, history, base_turn, on_token, thread_id=tid)
    finally:
        runtime_model.reset(tok_m)
        runtime_on_token.reset(tok_t)
        await maybe_release_after_run(tid)
    if not isinstance(result, dict):
        return await _fallback_send(model, tasks, history, base_turn, on_token, thread_id=tid)
    merged = merge_node(result)
    trace = dict(merged.get("trace") or {})
    bt = base_turn if isinstance(base_turn, dict) else {}
    if not trace.get("trace_id") and bt.get("trace_id"):
        trace["trace_id"] = str(bt.get("trace_id") or "")
    if companion_ckpt_snap is not None:
        trace["companion_checkpoint_snapshot"] = companion_ckpt_snap
    trace["engine"] = "langgraph"
    trace["pattern"] = "supervisor+transfer_to+review"
    trace["supervisor"] = True
    trace["agents"] = [major_to_agent_name(str(t.get("major") or "")) for t in tasks if isinstance(t, dict)]
    reply = str(merged.get("reply") or "").strip() or "我在，你可以继续说说近况。"
    worker_n = len([t for t in tasks if isinstance(t, dict)])
    emit_merge_reply_stream(reply, on_token, worker_count=worker_n)
    return (
        reply,
        list(result.get("facts") or []),
        trace,
    )


async def _fallback_send(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any,
    *,
    thread_id: str = "",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    from graph.companion import ainvoke_send_companion_graph

    return await ainvoke_send_companion_graph(
        model, tasks, history, base_turn, on_token=on_token, thread_id=thread_id
    )
