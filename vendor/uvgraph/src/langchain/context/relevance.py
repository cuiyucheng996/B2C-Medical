"""从 RouterState 构建 Agent 运行上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage

from ...algorithm.round1 import body_from_state, extract_query, latest_human_text
from ...knowledge.body_disease_map import BODY_PART_HINTS, BODY_PARTS

__all__ = [
    "RelevanceRunContext",
    "build_agent_messages",
    "build_run_context",
    "format_body_part_catalog",
    "format_recent_dialogue",
]


@dataclass(slots=True)
class RelevanceRunContext:
    """Agent 运行时上下文（经 middleware 读入/写出）。"""

    query: str
    existing_body_part: str | None = None
    recent_dialogue: str = ""
    body_part_catalog: str = ""
    turn_index: int = 1


def format_body_part_catalog() -> str:
    return "\n".join(f"- {part}：{BODY_PART_HINTS[part]}" for part in BODY_PARTS)


def format_recent_dialogue(messages: list[BaseMessage], *, limit: int = 6) -> str:
    if not messages:
        return "（无历史对话）"
    lines: list[str] = []
    for msg in messages[-limit:]:
        role = "用户" if msg.type == "human" else "助手"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        text = content.strip()
        if text:
            lines.append(f"{role}：{text}")
    return "\n".join(lines) if lines else "（无历史对话）"


def build_run_context(state: Mapping[str, Any]) -> RelevanceRunContext:
    messages = list(state.get("messages") or [])
    query = extract_query(state) or latest_human_text(state)
    return RelevanceRunContext(
        
        query=query.strip(),
        existing_body_part=body_from_state(state),
        recent_dialogue=format_recent_dialogue(messages),
        body_part_catalog=format_body_part_catalog(),
        turn_index=max(1, sum(1 for m in messages if m.type == "human")),
    )


def build_agent_messages(ctx: RelevanceRunContext) -> list[HumanMessage]:
    """组装送入 Agent 的用户消息（上下文 + 本轮输入）。"""
    body_hint = (
        f"已选定部位：{ctx.existing_body_part}"
        if ctx.existing_body_part
        else "已选定部位：无"
    )
    content = (
        f"【会话上下文】\n"
        f"轮次：第 {ctx.turn_index} 轮用户输入\n"
        f"{body_hint}\n\n"
        f"【近期对话】\n{ctx.recent_dialogue}\n\n"
        f"【八大部位参考】\n{ctx.body_part_catalog}\n\n"
        f"【本轮用户输入】\n{ctx.query or '（空）'}"
    )
    return [HumanMessage(content=content)]
