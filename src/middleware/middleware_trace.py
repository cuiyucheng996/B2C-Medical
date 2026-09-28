"""记录 create_agent 本轮中间件动作，供调试台展示。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from middleware.context import CompanionRunContext

__all__ = [
    "append_middleware_trace",
    "format_middleware_trace_body",
    "reset_middleware_trace",
    "take_middleware_trace",
    "trace_route_from_runtime",
]

_trace: ContextVar[list[dict[str, str]]] = ContextVar("companion_middleware_trace", default=[])


def reset_middleware_trace() -> None:
    _trace.set([])


def take_middleware_trace() -> list[dict[str, str]]:
    return list(_trace.get() or [])


def trace_route_from_runtime(runtime: Any) -> str:
    ctx = getattr(runtime, "context", None)
    if not isinstance(ctx, CompanionRunContext):
        return ""
    return " / ".join(part for part in (ctx.major, ctx.intent) if part)


def append_middleware_trace(name: str, detail: str, *, route: str = "", runtime: Any = None) -> None:
    label = (name or "").strip()
    if not label:
        return
    row_route = (route or "").strip() or trace_route_from_runtime(runtime)
    rows = list(_trace.get() or [])
    rows.append(
        {
            "name": label,
            "detail": (detail or "").strip()[:800],
            "route": row_route,
        }
    )
    _trace.set(rows)
    from algorithm.stream_debug import emit_stream_debug

    emit_stream_debug(
        {
            "kind": "middleware",
            "title": "中间件的使用",
            "name": label,
            "route": row_route,
            "body": (detail or "").strip()[:800],
        }
    )


def format_middleware_trace_body(row: dict[str, Any]) -> str:
    name = str(row.get("name") or "中间件")
    detail = str(row.get("detail") or "").strip() or "（无说明）"
    route = str(row.get("route") or "").strip()
    lines = [name]
    if route:
        lines.append(f"路线：{route}")
    lines.append(detail)
    return "\n".join(lines)
