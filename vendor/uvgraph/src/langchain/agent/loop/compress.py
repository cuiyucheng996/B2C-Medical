"""对话窗口压缩：旧轮收成摘要，长 ToolMessage / 过早回复截断。

不碰 Redis / Neo4j；只改即将送给模型的 messages。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

__all__ = [
    "KEEP_USER_TURNS",
    "MAX_CONTEXT_CHARS",
    "MAX_TOOL_CHARS",
    "compress_messages",
    "messages_need_compression",
]

MAX_CONTEXT_CHARS = 8000
MAX_TOOL_CHARS = 600
MAX_AI_CHARS = 1500
KEEP_USER_TURNS = 3
_SUMMARY_LINE = 80
_SUMMARY_CAP = 1200


def _text(msg: BaseMessage) -> str:
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= n:
        return text
    return text[: max(n - 1, 1)] + "…"


def _chars(messages: list[BaseMessage]) -> int:
    return sum(len(_text(m)) for m in messages)


def messages_need_compression(
    messages: list[BaseMessage],
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    max_tool_chars: int = MAX_TOOL_CHARS,
) -> bool:
    if _chars(messages) > max_chars:
        return True
    for msg in messages:
        if isinstance(msg, ToolMessage) and len(_text(msg)) > max_tool_chars:
            return True
        if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
            if len(_text(msg)) > MAX_AI_CHARS:
                return True
    return False


def _tool_names(msg: AIMessage) -> str:
    names: list[str] = []
    for call in getattr(msg, "tool_calls", None) or []:
        if isinstance(call, dict):
            names.append(str(call.get("name") or "tool"))
        else:
            names.append(str(getattr(call, "name", None) or "tool"))
    return "、".join(names)


def _summarize_old(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户：{_clip(_text(msg), _SUMMARY_LINE)}")
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                extra = _clip(_text(msg), 40)
                call = _tool_names(msg)
                bit = f"助手：调用{call}"
                if extra:
                    bit += f"（{extra}）"
                lines.append(bit)
            else:
                lines.append(f"助手：{_clip(_text(msg), _SUMMARY_LINE)}")
        elif isinstance(msg, ToolMessage):
            name = str(getattr(msg, "name", None) or "工具")
            lines.append(f"{name}：{_clip(_text(msg), 60)}")
        elif isinstance(msg, SystemMessage):
            continue
    body = "\n".join(lines)
    if len(body) > _SUMMARY_CAP:
        body = body[: _SUMMARY_CAP - 1] + "…"
    return "[历史摘要]\n" + (body or "（较早对话已省略）")


def _shrink_recent(msg: BaseMessage) -> BaseMessage:
    if isinstance(msg, ToolMessage) and len(_text(msg)) > MAX_TOOL_CHARS:
        return ToolMessage(
            content=_clip(_text(msg), MAX_TOOL_CHARS) + "[已截断]",
            tool_call_id=msg.tool_call_id,
            name=msg.name,
        )
    if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
        raw = _text(msg)
        if len(raw) > MAX_AI_CHARS:
            return AIMessage(content=_clip(raw, MAX_AI_CHARS))
    return msg


def compress_messages(
    messages: list[BaseMessage],
    *,
    keep_user_turns: int = KEEP_USER_TURNS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[BaseMessage]:
    """从最近一次用户发言起保留若干轮；更早的压成一条摘要。"""
    if not messages:
        return []
    if not messages_need_compression(messages, max_chars=max_chars):
        return list(messages)

    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    keep = max(1, keep_user_turns)
    if len(human_idx) <= keep:
        return [_shrink_recent(m) for m in messages]

    cut = human_idx[-keep]
    old, recent = messages[:cut], messages[cut:]
    summary = SystemMessage(content=_summarize_old(old))
    return [summary, *[_shrink_recent(m) for m in recent]]
