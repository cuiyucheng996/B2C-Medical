"""HITL 待办：全部落在 LangGraph checkpointer（与 create_agent 共用存储）。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from configs.paths_b2c import B2C_ROOT

logger = logging.getLogger(__name__)

HITL_KIND_SLOT = "slot_fill"
HITL_KIND_TOOL = "tool_approval"

__all__ = [
    "HITL_KIND_SLOT",
    "HITL_KIND_TOOL",
    "clear_session_hitl",
    "consume_slot_hitl",
    "get_hitl_checkpointer",
    "get_pending_hitl",
    "put_slot_pending",
    "put_tool_pending",
    "read_tool_pending",
    "pop_tool_pending",
    "slot_thread_id",
]


class _SlotState(TypedDict, total=False):
    pending: dict[str, Any]
    merged_query: str


class _MetaState(TypedDict, total=False):
    row: dict[str, Any]


def slot_thread_id(session_id: str) -> str:
    return f"hitl:slot:{(session_id or '').strip()}"


def meta_thread_id(session_id: str) -> str:
    return f"hitl:meta:{(session_id or '').strip()}"


@lru_cache(maxsize=1)
def get_hitl_checkpointer() -> Any:
    """与 create_agent 共用；默认 InMemorySaver（可装 langgraph-checkpoint-sqlite 后扩展落盘）。"""
    import os

    mode = (os.getenv("COMPANION_HITL_CHECKPOINTER") or "memory").strip().lower()
    if mode in {"sqlite", "disk", "file"}:
        path = B2C_ROOT / "data" / "companion_hitl_checkpoints.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            saver = SqliteSaver.from_conn_string(str(path))
            if hasattr(saver, "__enter__"):
                logger.warning("SqliteSaver 需上下文管理，HITL 回退 InMemorySaver")
            else:
                return saver
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite 未安装，HITL 使用 InMemorySaver")
    return InMemorySaver()


@lru_cache(maxsize=1)
def _slot_graph() -> Any:
    def wait_node(state: _SlotState) -> dict[str, Any]:
        pending = dict(state.get("pending") or {})
        answer = interrupt(pending)
        text = str(answer or "").strip()
        orig = str((pending.get("state") or {}).get("query") or "").strip()
        merged = f"{orig}。{text}" if orig and text else (orig or text)
        return {"merged_query": merged, "pending": {}}

    g = StateGraph(_SlotState)
    g.add_node("wait", wait_node)
    g.add_edge(START, "wait")
    g.add_edge("wait", END)
    return g.compile(checkpointer=get_hitl_checkpointer())


@lru_cache(maxsize=1)
def _meta_graph() -> Any:
    def passthrough(state: _MetaState) -> dict[str, Any]:
        return {"row": dict(state.get("row") or {})}

    g = StateGraph(_MetaState)
    g.add_node("meta", passthrough)
    g.add_edge(START, "meta")
    g.add_edge("meta", END)
    return g.compile(checkpointer=get_hitl_checkpointer())


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": (thread_id or "").strip()}}


def _delete_thread(graph: Any, thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    try:
        if hasattr(graph, "delete_thread"):
            graph.delete_thread(tid)
        elif hasattr(graph, "checkpointer") and graph.checkpointer is not None:
            cp = graph.checkpointer
            if hasattr(cp, "delete_thread"):
                cp.delete_thread(tid)
    except Exception:
        logger.debug("delete_thread failed thread_id=%s", tid, exc_info=True)


def _pending_row_from_slot_payload(
    session_id: str,
    payload: dict[str, Any],
    *,
    question: str,
    agent_name: str,
) -> dict[str, Any]:
    tid = slot_thread_id(session_id)
    return {
        "resume_token": tid,
        "thread_id": tid,
        "pending": True,
        "hitl_kind": HITL_KIND_SLOT,
        "question": question,
        "agent_name": agent_name,
        "state": dict(payload.get("state") or {}),
        "action_requests": [],
        "review_configs": [],
        "session_id": session_id,
    }


def _slot_interrupt_payload(session_id: str) -> dict[str, Any] | None:
    graph = _slot_graph()
    snap = graph.get_state(_config(slot_thread_id(session_id)))
    if not snap or not snap.interrupts:
        return None
    value = snap.interrupts[0].value
    return value if isinstance(value, dict) else None


def put_slot_pending(
    session_id: str,
    *,
    question: str,
    agent_name: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    sid = (session_id or "").strip()
    payload = {
        "question": question,
        "agent_name": agent_name,
        "state": dict(state or {}),
        "hitl_kind": HITL_KIND_SLOT,
    }
    graph = _slot_graph()
    tid = slot_thread_id(sid)
    graph.invoke({"pending": payload}, config=_config(tid))
    return _pending_row_from_slot_payload(
        sid, payload, question=question, agent_name=agent_name
    )


def consume_slot_hitl(session_id: str, resume_token: str, user_text: str) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    tid = slot_thread_id(sid)
    token = (resume_token or "").strip()
    if token and token != tid:
        return None
    payload = _slot_interrupt_payload(sid)
    if not payload:
        return None
    if not token:
        return _pending_row_from_slot_payload(
            sid,
            payload,
            question=str(payload.get("question") or ""),
            agent_name=str(payload.get("agent_name") or ""),
        )
    graph = _slot_graph()
    result = graph.invoke(Command(resume=user_text), config=_config(tid))
    merged = str((result or {}).get("merged_query") or user_text).strip()
    st = dict(payload.get("state") or {})
    st["merged_query"] = merged
    st["answer"] = user_text
    _delete_thread(graph, tid)
    row = _pending_row_from_slot_payload(
        sid,
        payload,
        question=str(payload.get("question") or ""),
        agent_name=str(payload.get("agent_name") or ""),
    )
    row["state"] = st
    return row


def put_tool_pending(session_id: str, row: dict[str, Any]) -> None:
    sid = (session_id or "").strip()
    _meta_graph().invoke({"row": dict(row)}, config=_config(meta_thread_id(sid)))


def read_tool_pending(session_id: str) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    snap = _meta_graph().get_state(_config(meta_thread_id(sid)))
    values = dict(snap.values or {}) if snap else {}
    row = values.get("row")
    if not isinstance(row, dict) or not row.get("pending"):
        return None
    return dict(row)


def pop_tool_pending(session_id: str) -> dict[str, Any] | None:
    row = read_tool_pending(session_id)
    if row:
        clear_tool_pending(session_id)
    return row


def clear_tool_pending(session_id: str) -> None:
    _delete_thread(_meta_graph(), meta_thread_id(session_id))


def get_pending_hitl(session_id: str) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    if not sid:
        return None
    payload = _slot_interrupt_payload(sid)
    if payload:
        return _pending_row_from_slot_payload(
            sid,
            payload,
            question=str(payload.get("question") or ""),
            agent_name=str(payload.get("agent_name") or ""),
        )
    return read_tool_pending(sid)


def clear_session_hitl(session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    _delete_thread(_slot_graph(), slot_thread_id(sid))
    _delete_thread(_meta_graph(), meta_thread_id(sid))
