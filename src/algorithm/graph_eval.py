"""评测台：完整陪聊图与意图子图的结构与一次试跑。"""

from __future__ import annotations

import time
import uuid
from typing import Any

__all__ = ["describe_graphs", "run_graph_probe"]


def _edges(compiled: Any) -> list[dict[str, str]]:
    drawn = compiled.get_graph()
    rows: list[dict[str, str]] = []
    for edge in getattr(drawn, "edges", []) or []:
        rows.append(
            {
                "source": str(getattr(edge, "source", "") or ""),
                "target": str(getattr(edge, "target", "") or ""),
            }
        )
    return rows


def _nodes(compiled: Any) -> list[str]:
    drawn = compiled.get_graph()
    names = [str(name) for name in (getattr(drawn, "nodes", {}) or {})]
    return [name for name in names if name not in ("__start__", "__end__")]


def describe_graphs() -> dict[str, Any]:
    from graph.companion import build_companion_graph
    from graph.intent_worker import build_intent_worker_graph
    from graph.supervisor_route import companion_use_supervisor

    full = build_companion_graph()
    sub = build_intent_worker_graph()
    return {
        "ok": True,
        "supervisor": companion_use_supervisor(),
        "full": {
            "name": "companion",
            "entry": "ainvoke_companion_intent_graph",
            "nodes": _nodes(full),
            "edges": _edges(full),
            "note": "begin_turn 后按 tasks Send 到 intent_worker；评审后可 peer_handoff 再进子图，最后 merge。",
        },
        "subgraph": {
            "name": "intent_worker",
            "nodes": _nodes(sub),
            "edges": _edges(sub),
            "note": "execute 调 create_agent；draft_review=off 时跳过评审 LLM；all/public 时 review 不通过且未超轮次则回到 execute。",
        },
    }


def _task_brief(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "major": str(item.get("major") or ""),
                "intent": str(item.get("intent") or ""),
                "handler": str(item.get("handler") or ""),
            }
        )
    return rows


async def run_graph_probe(text: str, *, mode: str = "full") -> dict[str, Any]:
    """mode=full 跑当前配置的完整图；mode=subgraph 只跑第一条意图的 intent_worker。"""
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty"}
    which = "subgraph" if (mode or "").strip() == "subgraph" else "full"
    from agent.companion_agent import _chat_model
    from algorithm.companion_turn import resolve_companion_turn

    model = _chat_model()
    if model is None:
        return {"ok": False, "reason": "未配置聊天模型", "structure": describe_graphs()}
    started = time.perf_counter()
    turned = resolve_companion_turn(raw, model)
    if not turned.get("ok"):
        return {"ok": False, "reason": turned.get("reason") or "preprocess_failed", "structure": describe_graphs()}
    tasks = [item for item in list(turned.get("tasks") or []) if isinstance(item, dict)]
    base_turn = {
        "query": str(turned.get("text") or raw),
        "session_id": f"eval-graph-{uuid.uuid4().hex[:12]}",
        "user_id": "eval-graph",
        "handler": str((tasks[0] or {}).get("handler") or ""),
    }
    observe = str(turned.get("observe") or "")
    detail: dict[str, Any] = {
        "ok": True,
        "mode": which,
        "query": base_turn["query"],
        "tasks": _task_brief(tasks),
        "structure": describe_graphs(),
    }
    try:
        if which == "subgraph":
            from graph.intent_worker import build_intent_worker_graph, runtime_model, runtime_on_token

            task = tasks[0] if tasks else {"major": "", "intent": "", "handler": "", "observe": observe}
            payload = {
                "major": str(task.get("major") or ""),
                "intent": str(task.get("intent") or ""),
                "handler": str(task.get("handler") or ""),
                "observe": str(task.get("observe") or observe),
            }
            state = {
                "query": base_turn["query"],
                "observe": payload["observe"],
                "history": [],
                "base_turn": base_turn,
                "task": payload,
                "review_hint": "",
                "review_round": 0,
                "draft": "",
                "facts": [],
                "review_pass": False,
                "review_reason": "",
                "worker_outputs": [],
            }
            tok_m = runtime_model.set(model)
            tok_t = runtime_on_token.set(None)
            try:
                result = await build_intent_worker_graph().ainvoke(state)
            finally:
                runtime_model.reset(tok_m)
                runtime_on_token.reset(tok_t)
            outputs = list((result or {}).get("worker_outputs") or []) if isinstance(result, dict) else []
            first = outputs[0] if outputs else {}
            detail["reply"] = str(first.get("reply") or "")
            detail["review"] = {
                "pass": bool(first.get("review_pass")),
                "reason": str(first.get("review_reason") or ""),
                "retries": int(first.get("review_retries") or 0),
            }
            detail["facts"] = list((result or {}).get("facts") or []) if isinstance(result, dict) else []
        else:
            from graph.companion import ainvoke_companion_intent_graph

            reply, facts, trace = await ainvoke_companion_intent_graph(
                model, tasks, [], base_turn, thread_id=base_turn["session_id"]
            )
            detail["reply"] = reply
            detail["facts"] = facts
            detail["trace"] = trace
    except Exception as exc:
        detail["ok"] = False
        detail["error"] = str(exc)
    detail["elapsed_s"] = round(time.perf_counter() - started, 2)
    return detail
