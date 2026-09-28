"""create_agent 与 HITL 共用 checkpointer。"""

from __future__ import annotations

from typing import Any

__all__ = ["get_agent_checkpointer"]


def get_agent_checkpointer() -> Any:
    from algorithm.hitl_checkpoint import get_hitl_checkpointer

    return get_hitl_checkpointer()
