"""每轮 ainvoke 入口：重置当轮 DAG 字段（跨轮记忆走 Redis/Graphiti，不靠 checkpoint）。"""

from __future__ import annotations

from typing import Any

from langgraph.types import Overwrite

from graph.state import CompanionGraphState

__all__ = ["begin_turn_node"]


def begin_turn_node(state: CompanionGraphState) -> dict[str, Any]:
    prev = int(state.get("turn_count") or 0)
    return {
        "worker_outputs": Overwrite([]),
        "facts": Overwrite([]),
        "wave1_outputs": [],
        "handoff_round": 0,
        "reply": "",
        "turn_count": prev + 1,
    }
