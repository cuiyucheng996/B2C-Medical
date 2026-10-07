"""本轮调试 SSE：流式推送 LLM / trace（tool_calls、CompanionToolEnvelope）。"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Callable

_emitter: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar(
    "companion_stream_debug_emitter",
    default=None,
)

__all__ = [
    "bind_stream_debug_emitter",
    "emit_checkpoint_version",
    "emit_rag_flow",
    "emit_stream_debug",
    "emit_trace_envelope",
    "emit_trace_tool_call_delta",
    "emit_trace_tool_calls",
    "reset_stream_debug_emitter",
    "unbind_stream_debug_emitter",
]


def bind_stream_debug_emitter(fn: Callable[[dict[str, Any]], None]) -> object:
    return _emitter.set(fn)


def unbind_stream_debug_emitter(token: object) -> None:
    _emitter.reset(token)


def reset_stream_debug_emitter() -> None:
    _emitter.set(None)


def emit_stream_debug(payload: dict[str, Any]) -> None:
    cb = _emitter.get()
    if cb is None:
        return
    row = dict(payload or {})
    if not row.get("kind"):
        return
    try:
        cb(row)
    except Exception:
        pass


def _trace_title(route: str, *, suffix: str) -> str:
    base = "子图 · " + suffix
    route = (route or "").strip()
    return f"{base} · {route}" if route else base


def emit_trace_tool_calls(tool_calls: list[Any], *, route: str = "") -> None:
    if not tool_calls:
        return
    emit_stream_debug(
        {
            "kind": "trace",
            "title": _trace_title(route, suffix="function call（tool_calls）"),
            "section": "tool_calls（LangChain 官方）",
            "body": json.dumps(tool_calls, ensure_ascii=False, indent=2, default=str),
        }
    )


def emit_trace_tool_call_delta(name: str, args_delta: str, *, route: str = "") -> None:
    if not (name or args_delta):
        return
    emit_stream_debug(
        {
            "kind": "trace",
            "title": _trace_title(route, suffix=f"function call 流式 · {name}"),
            "section": "tool_call_chunks",
            "body": args_delta,
            "append": True,
        }
    )


def emit_checkpoint_version(version: dict[str, Any]) -> None:
    """主图 checkpointer 每出现新版本时推送到调试 SSE（kind=checkpoint）。"""
    row = dict(version or {})
    if not row:
        return
    seq = row.get("seq")
    title = f"第{seq}版" if seq is not None else "checkpoint 版本"
    hint = str(row.get("node_hint") or "").strip()
    if hint:
        title += " · " + hint
    snap = row.get("channel_snapshot") or row.get("values_preview") or {}
    emit_stream_debug(
        {
            "kind": "checkpoint",
            "title": title,
            "version": row,
            "sections": [
                {
                    "title": "状态快照",
                    "body": json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                }
            ],
        }
    )


def emit_rag_flow(step: str, section: str, body: str) -> None:
    """陪聊调试台 RAG 增强：按步骤流式推送（前端可折叠标题）。"""
    title = (step or "RAG增强").strip()
    sec = (section or "步骤").strip()
    emit_stream_debug(
        {
            "kind": "rag",
            "step": title,
            "sections": [{"title": sec, "body": (body or "").strip() or "—"}],
        }
    )


def emit_trace_envelope(tool_name: str, body: str, *, route: str = "") -> None:
    emit_stream_debug(
        {
            "kind": "trace",
            "title": _trace_title(route, suffix=f"工具 · {tool_name or '?'}"),
            "section": "CompanionToolEnvelope",
            "body": (body or "").strip() or "（空）",
        }
    )
