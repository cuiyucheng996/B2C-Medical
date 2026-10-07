"""C 端 create_agent 运行时上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CompanionRunContext"]


@dataclass
class CompanionRunContext:
    query: str = ""
    observe: str = ""
    major: str = ""
    intent: str = ""
    handler: str = ""
    user_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    think_enabled: bool = False
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    # 主图并行子任务序号（build_intent_tasks）；本子 Agent 内第几次 model 循环
    task_index: int = 0
    task_total: int = 0
    model_call_seq: int = 0
    # 本子 Agent 第一次 model 时的 messages 条数（含历史+本轮 user+预处理观察），后续循环只缓存前 N 条
    prefix_message_count: int = 0
