"""陪聊运行时记忆对齐（单一关场/新开入口）。

Redis 键约定：
- conversation:short:{session_id} — 本场全量多轮（B2C）
- conversation:compress:{session_id} — 本场增量摘要
- conversation:slots:{user_id} — 槽位 JSON（含 session_id 字段校验）
- LangGraph checkpoint thread_id — 默认 base_turn.session_id；子 Agent 为 agent:{session_id}:{major}:{handler}
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "memory_scope_dict",
    "release_session_runtime_state",
    "validate_slot_memory_for_session",
]


def memory_scope_dict(session: Any) -> dict[str, str]:
    """调试/日志：本场 memory 作用域。"""
    uid = str(getattr(session, "user_id", "") or getattr(session, "group_id", "") or "").strip()
    sid = str(getattr(session, "session_id", "") or "").strip()
    return {
        "user_id": uid,
        "session_id": sid,
        "short_redis": f"conversation:short:{sid}" if sid else "",
        "compress_redis": f"conversation:compress:{sid}" if sid else "",
        "slot_redis": f"conversation:slots:{uid}" if uid else "",
        "graph_thread_id": sid,
    }


def validate_slot_memory_for_session(
    slots: dict[str, list[str]],
    *,
    user_id: str,
    session_id: str,
) -> dict[str, list[str]]:
    """槽位按 user 存 Redis；仅当 payload.session_id 与当前场一致才恢复。"""
    uid = (user_id or "").strip()
    sid = (session_id or "").strip()
    if not uid or not sid:
        return slots
    from algorithm.short_memory import load_b2c_slot_memory

    loaded = load_b2c_slot_memory(uid, session_id=sid)
    if any(loaded.get(key) for key in ("疾病", "症状", "部位")):
        return loaded
    return slots


def release_session_runtime_state(
    user_id: str,
    session_id: str,
    *,
    clear_slots: bool = True,
    clear_checkpoint: bool = True,
) -> None:
    """关场 / 新开替换：清本场 short+compress；可选清槽位与 LangGraph/HITL checkpoint。"""
    uid = (user_id or "").strip()
    sid = (session_id or "").strip()
    if not sid and not uid:
        return
    from algorithm.short_memory import clear_b2c_short_memory

    clear_b2c_short_memory(uid, session_id=sid, clear_slots=clear_slots)
    if clear_checkpoint and sid:
        from graph.checkpoint_lifecycle import release_on_session_end

        release_on_session_end(sid)
