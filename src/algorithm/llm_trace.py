"""本轮陪聊每次 LLM invoke 的完整 messages 采集（调试台「各环节 LLM 调用上下文组成」）。"""

from __future__ import annotations

import json
import threading
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "format_messages_for_display",
    "llm_trace_to_briefs",
    "record_llm_call",
    "note_token_usage",
    "reset_llm_trace",
    "turn_token_footer",
    "serialize_langchain_messages",
    "take_llm_trace",
]

_token_usage: dict[str, dict[str, int]] = {}
_trace: ContextVar[list[dict[str, Any]] | None] = ContextVar("companion_llm_trace", default=None)
_seq: ContextVar[int] = ContextVar("companion_llm_trace_seq", default=0)
_turn_traces: dict[str, list[dict[str, Any]]] = {}
_turn_seq: dict[str, int] = {}
_bucket_lock = threading.Lock()

# 七类固定小标题（与产品约定一致）
STAGE_SPELL = "纠错 Agent"
STAGE_COREF = "指代补全"
STAGE_INTENT_MAIN = "意图主判"
STAGE_INTENT_FINE = "意图补全/多路"
STAGE_AGENT = "各路执行"
STAGE_THINK = "思考模式"
STAGE_REVIEW = "评审"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_SPELL,
    STAGE_COREF,
    STAGE_INTENT_MAIN,
    STAGE_INTENT_FINE,
    STAGE_AGENT,
    STAGE_THINK,
    STAGE_REVIEW,
)


def _trace_key() -> str:
    from algorithm.run_trace import current_trace_id

    return (current_trace_id() or "").strip() or "_turn"


def _active_bucket() -> list[dict[str, Any]]:
    """按 trace_id 汇总；LangGraph 并行 worker 与主线程共用同一列表。"""
    key = _trace_key()
    with _bucket_lock:
        rows = _turn_traces.get(key)
        if rows is None:
            rows = []
            _turn_traces[key] = rows
    _trace.set(rows)
    return rows


def reset_llm_trace() -> None:
    key = _trace_key()
    with _bucket_lock:
        _turn_traces[key] = []
        _turn_seq[key] = 0
    _trace.set(_turn_traces[key])
    _seq.set(0)
    with _bucket_lock:
        _token_usage[key] = {"input": 0, "output": 0, "total": 0}


def note_token_usage(message: Any) -> None:
    """累加本轮模型返回里的 token。没有用量字段则跳过。"""
    usage = getattr(message, "usage_metadata", None)
    if usage is not None and not isinstance(usage, dict):
        usage = {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    if not isinstance(usage, dict):
        meta = getattr(message, "response_metadata", None)
        token_usage = meta.get("token_usage") if isinstance(meta, dict) else None
        usage = token_usage if isinstance(token_usage, dict) else None
    if not isinstance(usage, dict):
        return
    inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (inp + out))
    if total <= 0 and inp <= 0 and out <= 0:
        return
    key = _trace_key()
    with _bucket_lock:
        bucket = _token_usage.setdefault(key, {"input": 0, "output": 0, "total": 0})
        bucket["input"] += inp
        bucket["output"] += out
        bucket["total"] += total or (inp + out)


def turn_token_footer(reply: str) -> str:
    key = _trace_key()
    with _bucket_lock:
        bucket = dict(_token_usage.get(key) or {})
    total = int(bucket.get("total") or 0)
    if total <= 0:
        return reply
    line = (
        f"（本轮消耗 token：输入 {int(bucket.get('input') or 0)}，"
        f"输出 {int(bucket.get('output') or 0)}，合计 {total}）"
    )
    if line in (reply or ""):
        return reply
    body = (reply or "").rstrip()
    return f"{body}\n\n{line}" if body else line


def take_llm_trace() -> list[dict[str, Any]]:
    key = _trace_key()
    with _bucket_lock:
        rows = list(_turn_traces.get(key) or [])
    if rows:
        return rows
    cur = _trace.get()
    return list(cur) if cur is not None else []


def run_in_llm_trace_context(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    """ThreadPool 内调用 LLM 时继承当前轮的 trace 列表。"""
    from contextvars import copy_context

    return copy_context().run(func, *args, **kwargs)


def serialize_langchain_messages(messages: list[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages or []:
        if isinstance(msg, SystemMessage):
            out.append({"role": "system", "content": _content_str(msg)})
        elif isinstance(msg, HumanMessage):
            out.append({"role": "user", "content": _content_str(msg)})
        elif isinstance(msg, AIMessage):
            out.append({"role": "assistant", "content": _content_str(msg)})
        elif isinstance(msg, ToolMessage):
            name = str(getattr(msg, "name", "") or "")
            out.append({"role": "tool", "name": name, "content": _content_str(msg)})
        elif isinstance(msg, BaseMessage):
            out.append({"role": str(getattr(msg, "type", "message")), "content": _content_str(msg)})
        elif isinstance(msg, dict):
            out.append(
                {
                    "role": str(msg.get("role") or "message"),
                    "content": str(msg.get("content") or ""),
                }
            )
    return out


def _content_str(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            return str(content)
    return str(content)


def format_messages_for_display(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for idx, row in enumerate(messages or [], start=1):
        role = str(row.get("role") or "message")
        name = str(row.get("name") or "").strip()
        tag = f"{role}" + (f" ({name})" if name else "")
        parts.append(f"--- [{idx}] {tag} ---\n{str(row.get('content') or '')}")
    return "\n\n".join(parts) if parts else "（无 messages）"


def record_llm_call(
    stage: str,
    messages: list[dict[str, str]] | list[Any],
    *,
    response: str = "",
    label: str = "",
) -> None:
    if messages and messages[0] and not isinstance(messages[0], dict):
        packed = serialize_langchain_messages(list(messages))
    else:
        packed = [dict(row) for row in (messages or []) if isinstance(row, dict)]
    key = _trace_key()
    with _bucket_lock:
        seq = int(_turn_seq.get(key, 0)) + 1
        _turn_seq[key] = seq
    _seq.set(seq)
    rows = _active_bucket()
    entry = {
        "seq": seq,
        "stage": (stage or "").strip() or "LLM",
        "label": (label or "").strip(),
        "messages": packed,
        "response": (response or "").strip(),
    }
    rows.append(entry)
    try:
        from algorithm.stream_debug import emit_stream_debug

        stage_s = str(entry["stage"])
        label_s = str(entry["label"] or "").strip()
        title = f"{stage_s}" + (f" · {label_s}" if label_s else "")
        body = format_messages_for_display(packed)
        resp = str(entry.get("response") or "").strip()
        if resp:
            body = body + "\n\n======== 模型输出 ========\n" + resp
        emit_stream_debug(
            {
                "kind": "llm",
                "step": title,
                "sections": [{"title": "完整上下文（送入模型的 messages）", "body": body}],
                "got": resp[:500] if resp else "（见上文 messages / 输出）",
            }
        )
    except Exception:
        pass


def _row_to_brief(row: dict[str, Any]) -> dict[str, Any]:
    stage = str(row.get("stage") or "LLM")
    label = str(row.get("label") or "").strip()
    title = f"{stage}" + (f" · {label}" if label else "")
    body = format_messages_for_display(list(row.get("messages") or []))
    resp = str(row.get("response") or "").strip()
    if resp:
        body = body + "\n\n======== 模型输出 ========\n" + resp
    return {
        "step": title,
        "sections": [{"title": "完整上下文（送入模型的 messages）", "body": body}],
        "got": resp[:500] if resp else "（见上文 messages / 输出）",
    }


def llm_trace_to_briefs(*, include_empty_stages: bool = True) -> list[dict[str, Any]]:
    """供 companion 调试台 paintLlm：按七类小标题顺序展示，每类可多条。"""
    rows = take_llm_trace()
    by_stage: dict[str, list[dict[str, Any]]] = {s: [] for s in STAGE_ORDER}
    other: list[dict[str, Any]] = []
    for row in rows:
        stage = str(row.get("stage") or "LLM")
        if stage in by_stage:
            by_stage[stage].append(row)
        else:
            other.append(row)
    steps: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        bucket = by_stage.get(stage) or []
        if bucket:
            for row in sorted(bucket, key=lambda r: int(r.get("seq") or 0)):
                steps.append(_row_to_brief(row))
        elif include_empty_stages:
            steps.append(
                {
                    "step": stage,
                    "sections": [
                        {
                            "title": "完整上下文（送入模型的 messages）",
                            "body": "（本轮该环节未调用 LLM，或走 BERT/规则短路）",
                        }
                    ],
                    "got": "—",
                }
            )
    for row in sorted(other, key=lambda r: int(r.get("seq") or 0)):
        steps.append(_row_to_brief(row))
    return steps
