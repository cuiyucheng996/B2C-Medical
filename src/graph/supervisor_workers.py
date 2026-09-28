"""按 major 注册的子智能体：内部仍走 execute→review 子图。"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from graph.intent_worker import run_intent_worker
from graph.supervisor_state import CompanionSupervisorState, major_to_agent_name

logger = logging.getLogger(__name__)

__all__ = ["build_major_worker_graph", "major_worker_graphs_for_tasks"]


def _task_payload_fields(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "major": str(task.get("major") or ""),
        "intent": str(task.get("intent") or ""),
        "handler": str(task.get("handler") or ""),
        "observe": str(task.get("observe") or ""),
    }


async def _run_major_worker(state: CompanionSupervisorState, *, major: str) -> dict[str, Any]:
    ctx = state.get("companion_ctx") if isinstance(state.get("companion_ctx"), dict) else {}
    tasks = [t for t in list(ctx.get("tasks") or []) if isinstance(t, dict) and str(t.get("major") or "") == major]
    if not tasks:
        return {
            "messages": [AIMessage(content=f"（{major}）无待办任务", name=major_to_agent_name(major))],
        }
    outputs: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    query = str(ctx.get("query") or "")
    observe = str(ctx.get("observe") or "")
    history = list(ctx.get("history") or [])
    base_turn = dict(ctx.get("base_turn") or {})
    for task in tasks:
        payload = _task_payload_fields(task)
        row = await run_intent_worker(
            {
                "query": query,
                "observe": payload["observe"] or observe,
                "history": history,
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
        )
        outputs.extend(list(row.get("worker_outputs") or []))
        facts.extend(list(row.get("facts") or []))
    text = "\n\n".join(str(item.get("reply") or "").strip() for item in outputs if str(item.get("reply") or "").strip())
    return {
        "messages": [AIMessage(content=text or "（子智能体无输出）", name=major_to_agent_name(major))],
        "worker_outputs": outputs,
        "facts": facts,
    }


@lru_cache(maxsize=16)
def build_major_worker_graph(major: str) -> Any:
    agent_name = major_to_agent_name(major)

    async def work(state: CompanionSupervisorState) -> dict[str, Any]:
        return await _run_major_worker(state, major=major)

    builder = StateGraph(CompanionSupervisorState)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile(name=agent_name)


def major_worker_graphs_for_tasks(tasks: list[dict[str, Any]]) -> list[Any]:
    majors = sorted({str(t.get("major") or "other") for t in tasks if isinstance(t, dict)})
    if not majors:
        majors = ["other"]
    return [build_major_worker_graph(major) for major in majors]
