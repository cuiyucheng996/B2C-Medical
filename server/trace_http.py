"""跨进程 HTTP 传递 trace_id（与 8002 algorithm.run_trace 对齐）。"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

logger = logging.getLogger("b2c.trace.http")

TRACE_HEADER = "X-Trace-Id"
_trace: ContextVar[str] = ContextVar("b2c_http_trace_id", default="")


def current_trace_id() -> str:
    return (_trace.get() or "").strip()


def bind_trace_id(trace_id: str) -> str:
    tid = (trace_id or "").strip()
    if not tid:
        tid = uuid.uuid4().hex[:16]
    _trace.set(tid)
    return tid


def ensure_trace_id() -> str:
    tid = current_trace_id()
    if tid:
        return tid
    return bind_trace_id("")


def inbound_trace_id(headers: dict[str, str] | None) -> str:
    if not headers:
        return ""
    for key in (TRACE_HEADER, "x-trace-id", "X-Trace-Id"):
        val = headers.get(key)
        if val:
            return str(val).strip()
    return ""


def internal_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {TRACE_HEADER: ensure_trace_id()}
    try:
        from algorithm.otel_setup import inject_trace_context

        inject_trace_context(headers)
    except Exception:
        pass
    if extra:
        headers.update(extra)
    return headers
