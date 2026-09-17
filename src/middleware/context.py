"""C 端 create_agent 运行时上下文。"""

from __future__ import annotations

from dataclasses import dataclass

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
    think_enabled: bool = False
