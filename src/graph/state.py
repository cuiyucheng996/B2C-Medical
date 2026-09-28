"""C 端陪聊 LangGraph 状态：共享字段 + reducer 合并并行子图结果。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

__all__ = ["CompanionGraphState", "IntentWorkerState", "merge_dicts"]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(left or {})
    out.update(right or {})
    return out


class IntentWorkerState(TypedDict, total=False):
    query: str
    observe: str
    peer_brief: str
    history: list[Any]
    base_turn: dict[str, Any]
    task: dict[str, Any]
    review_hint: str
    review_round: int
    draft: str
    facts: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    middleware_trace: list[dict[str, Any]]
    bound_tools: list[str]
    review_pass: bool
    review_reason: str
    worker_outputs: Annotated[list[dict[str, Any]], operator.add]


class CompanionGraphState(TypedDict, total=False):
    thread_id: str
    turn_count: int
    query: str
    observe: str
    history: list[Any]
    base_turn: dict[str, Any]
    tasks: list[dict[str, Any]]
    handoff_round: int
    wave1_outputs: list[dict[str, Any]]
    worker_outputs: Annotated[list[dict[str, Any]], operator.add]
    facts: Annotated[list[dict[str, Any]], operator.add]
    reply: str
    trace: Annotated[dict[str, Any], merge_dicts]
