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
