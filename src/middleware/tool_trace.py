"""记录 create_agent 本轮工具调用，供调试台展示 CompanionToolEnvelope。"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest

from tools.tool_envelope import is_envelope
from tools.mcp_bridge import tag_tool_trace_row

__all__ = [
    "CompanionToolTraceMiddleware",
    "format_tool_trace_step_body",
    "reset_tool_trace",
    "take_tool_trace",
    "tool_trace_from_messages",
]

_trace: ContextVar[list[dict[str, Any]]] = ContextVar("companion_tool_trace", default=[])

_ENVELOPE_BODY_LIMIT = 2400


def reset_tool_trace() -> None:
    _trace.set([])


def take_tool_trace() -> list[dict[str, Any]]:
    return list(_trace.get() or [])


def _parse_envelope(content: Any) -> dict[str, Any] | None:
    text = content if isinstance(content, str) else str(content or "")
    try:
        row = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not is_envelope(row):
        return None
    return {
        "ok": bool(row.get("ok")),
        "tool": str(row.get("tool") or ""),
        "usable": bool(row.get("usable")),
        "retryable": bool(row.get("retryable")),
        "error": str(row.get("error") or ""),
        "summary": str(row.get("summary") or ""),
        "data": dict(row.get("data") or {}) if isinstance(row.get("data"), dict) else {},
    }


def format_tool_trace_step_body(step: dict[str, Any]) -> str:
    """调试台：入参 + CompanionToolEnvelope 全文。"""
    head = ""
    if str(step.get("source") or "") == "mcp":
        srv = str(step.get("mcp_server") or "").strip()
        head = f"【MCP{' · ' + srv if srv else ''}】\n"
    args = str(step.get("args") or "").strip() or "（无）"
    lines = [head + f"入参：{args}" if head else f"入参：{args}"]
    env = step.get("envelope") if isinstance(step.get("envelope"), dict) else None
    if env:
        lines.append("— CompanionToolEnvelope —")
        lines.append(f"tool: {env.get('tool') or step.get('name') or '？'}")
        lines.append(f"ok: {env.get('ok')}  usable: {env.get('usable')}  retryable: {env.get('retryable')}")
        if str(env.get("error") or "").strip():
            lines.append(f"error: {env.get('error')}")
        if str(env.get("summary") or "").strip():
            lines.append(f"summary: {env.get('summary')}")
        data = env.get("data")
        if isinstance(data, dict) and data:
            blob = json.dumps(data, ensure_ascii=False, indent=2)
            if len(blob) > _ENVELOPE_BODY_LIMIT:
                blob = blob[:_ENVELOPE_BODY_LIMIT] + "\n…（data 已截断）"
            lines.append("data:\n" + blob)
        return "\n".join(lines)
    summary = str(step.get("summary") or "").strip()
    ok = str(step.get("ok") or "?")
    lines.append(f"回执（非标准信封）：{summary or '（空）'}")
    lines.append(f"成功标记：{ok}")
    raw = str(step.get("raw") or "").strip()
    if raw:
        lines.append(raw[:800])
    return "\n".join(lines)


def _append(
    *,
    name: str,
    args: str,
    summary: str,
    ok: str,
    envelope: dict[str, Any] | None,
    raw: str,
) -> None:
    rows = list(_trace.get() or [])
    row: dict[str, Any] = {
        "name": name,
        "args": (args or "").strip()[:300],
        "summary": (summary or "").strip()[:500],
        "ok": ok,
    }
    if envelope:
        row["envelope"] = envelope
    if raw:
        row["raw"] = raw[:1200]
    rows.append(row)
    _trace.set(rows)


def _summary_from_envelope(env: dict[str, Any]) -> tuple[str, str]:
    ok = "是" if env.get("ok") else "否"
    summary = str(env.get("summary") or env.get("error") or "").strip()
    data = env.get("data") if isinstance(env.get("data"), dict) else {}
    if not summary and data.get("facts"):
        summary = f"{len(data['facts'])} 条 facts"
    elif not summary and data.get("items"):
        summary = f"{len(data['items'])} 条结果"
    return summary or "（无摘要）", ok


def _row_from_call(request: ToolCallRequest, result: Any) -> dict[str, Any]:
    call = request.tool_call if isinstance(request.tool_call, dict) else {}
    name = str(call.get("name") or "")
    args = str(call.get("args") or "")
    content = getattr(result, "content", result) if result is not None else ""
    text = content if isinstance(content, str) else str(content or "")
    env = _parse_envelope(content)
    if env:
        summary, ok = _summary_from_envelope(env)
        return {
            "name": name,
            "args": (args or "").strip()[:300],
            "summary": (summary or "").strip()[:500],
            "ok": ok,
            "envelope": env,
            "raw": text[:1200],
        }
    return {
        "name": name,
        "args": (args or "").strip()[:300],
        "summary": text[:500],
        "ok": "?",
        "raw": text[:1200],
    }


def _ctx_tool_trace(request: ToolCallRequest) -> list[dict[str, Any]] | None:
    from middleware.context import CompanionRunContext

    runtime = getattr(request, "runtime", None)
    ctx = getattr(runtime, "context", None)
    if isinstance(ctx, CompanionRunContext) and isinstance(ctx.tool_trace, list):
        return ctx.tool_trace
    return None


def tool_trace_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """从 Agent 消息链兜底解析 ToolMessage（并行子图时避免 ContextVar 串线）。"""
    from langchain_core.messages import AIMessage, ToolMessage

    rows: list[dict[str, Any]] = []
    pending: dict[str, dict[str, str]] = {}
    for msg in messages or []:
        if isinstance(msg, AIMessage):
            for item in getattr(msg, "tool_calls", None) or []:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("id") or "")
                if tid:
                    pending[tid] = {
                        "name": str(item.get("name") or ""),
                        "args": str(item.get("args") or ""),
                    }
        if not isinstance(msg, ToolMessage):
            continue
        tid = str(getattr(msg, "tool_call_id", "") or "")
        meta = pending.get(tid) or {}
        name = str(getattr(msg, "name", None) or meta.get("name") or "")
        args = meta.get("args") or ""
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content or "")
        env = _parse_envelope(content)
        if env:
            summary, ok = _summary_from_envelope(env)
            rows.append(
                tag_tool_trace_row(
                    {
                        "name": name,
                        "args": (args or "").strip()[:300],
                        "summary": (summary or "").strip()[:500],
                        "ok": ok,
                        "envelope": env,
                        "raw": text[:1200],
                    }
                )
            )
        else:
            rows.append(
                tag_tool_trace_row(
                    {
                        "name": name,
                        "args": (args or "").strip()[:300],
                        "summary": text[:500],
                        "ok": "?",
                        "raw": text[:1200],
                    }
                )
            )
    return rows


class CompanionToolTraceMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        result = handler(request)
        _record(request, result)
        return result

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Any) -> Any:
        result = await handler(request)
        _record(request, result)
        return result


def _stamp_row(row: dict[str, Any], request: ToolCallRequest | None = None) -> dict[str, Any]:
    from algorithm.run_trace import current_trace_id, log_trace_event

    out = dict(row)
    tid = current_trace_id()
    if not tid and request is not None:
        from middleware.context import CompanionRunContext

        runtime = getattr(request, "runtime", None)
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, CompanionRunContext):
            tid = str(ctx.trace_id or "").strip()
    if tid:
        out["trace_id"] = tid
    out["ts"] = time.time()
    log_trace_event(
        "tool_call",
        tool=str(out.get("name") or ""),
        ok=str(out.get("ok") or ""),
        source=str(out.get("source") or "local"),
        mcp_server=str(out.get("mcp_server") or "") or None,
        summary=str(out.get("summary") or "")[:200] or None,
    )
    return out


def _record(request: ToolCallRequest, result: Any) -> None:
    row = _stamp_row(tag_tool_trace_row(_row_from_call(request, result)), request)
    sink = _ctx_tool_trace(request)
    if sink is not None:
        sink.append(row)
    else:
        _append(
            name=str(row.get("name") or ""),
            args=str(row.get("args") or ""),
            summary=str(row.get("summary") or ""),
            ok=str(row.get("ok") or "?"),
            envelope=row.get("envelope") if isinstance(row.get("envelope"), dict) else None,
            raw=str(row.get("raw") or ""),
        )
    try:
        from algorithm.stream_debug import emit_trace_envelope
        from middleware.context import CompanionRunContext

        route = ""
        runtime = getattr(request, "runtime", None)
        ctx = getattr(runtime, "context", None)
        if isinstance(ctx, CompanionRunContext):
            parts = [p for p in (ctx.major, ctx.intent, ctx.handler) if p]
            route = " / ".join(parts)
        emit_trace_envelope(
            str(row.get("name") or ""),
            format_tool_trace_step_body(row),
            route=route,
        )
    except Exception:
        pass
