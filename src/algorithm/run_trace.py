"""陪聊可观测：链路 trace_id + 结构化 JSON 日志（工具/MCP/图）。"""

from __future__ import annotations

import json
import logging
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

logger = logging.getLogger("b2c.trace")

_trace_id: ContextVar[str] = ContextVar("companion_trace_id", default="")

__all__ = [
    "bind_trace_id",
    "build_round_observability",
    "current_trace_id",
    "log_trace_event",
    "start_turn_trace",
]


def start_turn_trace(*, session_id: str = "", user_id: str = "", query: str = "") -> str:
    tid = uuid4().hex[:16]
    _trace_id.set(tid)
    preview = (query or "").strip()[:120]
    try:
        from algorithm.otel_setup import begin_turn_span

        begin_turn_span(
            tid,
            session_id=(session_id or "").strip(),
            user_id=(user_id or "").strip(),
            query_preview=preview,
        )
    except Exception:
        pass
    log_trace_event(
        "turn_start",
        session_id=(session_id or "").strip(),
        user_id=(user_id or "").strip(),
        query_preview=preview,
    )
    return tid


def bind_trace_id(trace_id: str) -> None:
    tid = (trace_id or "").strip()
    if tid:
        _trace_id.set(tid)
        try:
            from algorithm.otel_setup import bind_trace_attribute

            bind_trace_attribute(tid)
        except Exception:
            pass


def current_trace_id() -> str:
    return (_trace_id.get() or "").strip()


def log_trace_event(event: str, **fields: Any) -> None:
    tid = current_trace_id()
    payload: dict[str, Any] = {
        "event": (event or "").strip() or "unknown",
        "ts": time.time(),
    }
    if tid:
        payload["trace_id"] = tid
    for key, val in fields.items():
        if val is None or val == "":
            continue
        payload[key] = val
    try:
        from algorithm.otel_setup import emit_trace_event, end_turn_span

        emit_trace_event(event, **fields)
        if (event or "").strip() == "turn_end":
            end_turn_span(**{k: v for k, v in fields.items() if k in {"session_id", "tool_steps"}})
    except Exception:
        pass
    try:
        logger.info("%s", json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        logger.info("trace event=%s trace_id=%s", event, tid)


def build_round_observability(langgraph: dict[str, Any] | None) -> dict[str, Any]:
    g = langgraph if isinstance(langgraph, dict) else {}
    tid = str(g.get("trace_id") or current_trace_id() or "").strip()
    tool_runs = []
    for run in list(g.get("tool_runs") or []):
        if not isinstance(run, dict):
            continue
        steps = []
        for st in list(run.get("steps") or []):
            if not isinstance(st, dict):
                continue
            steps.append(
                {
                    "name": st.get("name"),
                    "ok": st.get("ok"),
                    "source": st.get("source"),
                    "mcp_server": st.get("mcp_server"),
                    "summary": st.get("summary"),
                    "args": st.get("args"),
                    "envelope": st.get("envelope") if isinstance(st.get("envelope"), dict) else None,
                }
            )
        mw = []
        for row in list(run.get("middleware_trace") or []):
            if isinstance(row, dict):
                mw.append({"name": row.get("name"), "route": row.get("route"), "detail": row.get("detail")})
        tool_runs.append(
            {
                "major": run.get("major"),
                "intent": run.get("intent"),
                "handler": run.get("handler"),
                "bound_tools": list(run.get("bound_tools") or []),
                "steps": steps,
                "middleware_trace": mw,
            }
        )
    return {
        "trace_id": tid,
        "engine": g.get("engine"),
        "pattern": g.get("pattern"),
        "thread_id": g.get("thread_id"),
        "workers": g.get("workers"),
        "peer_handoff": bool(g.get("peer_handoff")),
        "checkpointer": g.get("checkpointer"),
        "checkpoint_retention": g.get("checkpoint_retention"),
        "reviews": list(g.get("reviews") or []),
        "tool_runs": tool_runs,
    }
