"""陪聊工具 HITL：哪些 tool 需在执行前人工确认。"""

from __future__ import annotations

import os
from typing import Any

from langchain.agents.middleware import InterruptOnConfig

__all__ = [
    "companion_tool_hitl_enabled",
    "companion_tool_interrupt_on",
]

_TASK_TOOLS = (
    "note_task_remind",
    "note_task_checkin",
    "note_task_medication",
    "note_task_appointment",
    "note_task_report",
)

_DEFAULT_ALLOWED: list[str] = ["approve", "edit", "reject", "respond"]


def companion_tool_hitl_enabled() -> bool:
    raw = (os.getenv("COMPANION_TOOL_HITL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def companion_tool_interrupt_on() -> dict[str, bool | InterruptOnConfig]:
    """未列出的工具默认自动执行（HumanInTheLoopMiddleware 行为）。"""
    cfg: InterruptOnConfig = {
        "allowed_decisions": list(_DEFAULT_ALLOWED),
        "description": "具体任务将写入提醒/日程类记录，请确认参数是否正确。",
    }
    return {name: cfg for name in _TASK_TOOLS}
