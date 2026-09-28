"""多意图并行后的同伴 handoff：汇总初稿，二轮 Send 注入交叉上下文。"""

from __future__ import annotations

import os
from typing import Any

from langgraph.types import Overwrite, Send

from graph.state import CompanionGraphState

__all__ = [
    "companion_peer_handoff_enabled",
    "dispatch_handoff_intents",
    "format_peer_brief",
    "peer_handoff_node",
    "route_after_intent_worker",
]


def companion_peer_handoff_enabled() -> bool:
    """默认关：多意图并行一次出稿，由 merge 合并；硬依赖场景再设 COMPANION_PEER_HANDOFF=1。"""
    raw = (os.getenv("COMPANION_PEER_HANDOFF") or "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _task_key(task: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(task.get("major") or ""),
        str(task.get("intent") or ""),
        str(task.get("handler") or ""),
    )


def format_peer_brief(
    outputs: list[dict[str, Any]],
    *,
    exclude: tuple[str, str, str] | None = None,
    max_chars: int = 2400,
) -> str:
    lines: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("major") or ""),
            str(item.get("intent") or ""),
            str(item.get("handler") or ""),
        )
        if exclude and key == exclude:
            continue
        body = str(item.get("reply") or "").strip()
        if not body:
            continue
        lines.append(
            f"- [{key[0]} / {key[1]}] {body[:800]}{'…' if len(body) > 800 else ''}"
        )
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def route_after_intent_worker(state: CompanionGraphState) -> str:
    if int(state.get("handoff_round") or 0) >= 1:
        return "merge"
    tasks = [t for t in list(state.get("tasks") or []) if isinstance(t, dict)]
    if len(tasks) <= 1:
        return "merge"
    if not companion_peer_handoff_enabled():
        return "merge"
    outputs = [o for o in list(state.get("worker_outputs") or []) if isinstance(o, dict)]
    if not outputs:
        return "merge"
    return "peer_handoff"


def peer_handoff_node(state: CompanionGraphState) -> dict[str, Any]:
    outputs = [o for o in list(state.get("worker_outputs") or []) if isinstance(o, dict)]
    return {
        "handoff_round": 1,
        "wave1_outputs": outputs,
        "worker_outputs": Overwrite([]),
        "trace": {
            "peer_handoff": True,
            "wave1_workers": len(outputs),
        },
    }


def _worker_send_payload(
    state: CompanionGraphState,
    task: dict[str, Any],
    *,
    peer_brief: str,
) -> dict[str, Any]:
    payload = {
        "major": str(task.get("major") or ""),
        "intent": str(task.get("intent") or ""),
        "handler": str(task.get("handler") or ""),
        "observe": str(task.get("observe") or ""),
    }
    return {
        "query": str(state.get("query") or ""),
        "observe": payload["observe"] or str(state.get("observe") or ""),
        "history": list(state.get("history") or []),
        "base_turn": dict(state.get("base_turn") or {}),
        "task": payload,
        "peer_brief": peer_brief,
        "review_hint": "",
        "review_round": 0,
        "draft": "",
        "facts": [],
        "review_pass": False,
        "review_reason": "",
        "worker_outputs": [],
    }


def dispatch_handoff_intents(state: CompanionGraphState) -> list[Send]:
    tasks = [item for item in list(state.get("tasks") or []) if isinstance(item, dict)]
    wave1 = [o for o in list(state.get("wave1_outputs") or []) if isinstance(o, dict)]
    if not tasks:
        return [Send("merge", state)]
    sends: list[Send] = []
    for task in tasks:
        brief = format_peer_brief(wave1, exclude=_task_key(task))
        sends.append(
            Send(
                "intent_worker",
                _worker_send_payload(state, task, peer_brief=brief),
            )
        )
    return sends
