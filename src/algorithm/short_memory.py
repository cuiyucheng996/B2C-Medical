"""B2C 短期对话：Redis key 用 session_id（conversation:short:{session_id}），本场全量不滑动。"""

from __future__ import annotations

import logging
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "clear_b2c_short_memory",
    "load_b2c_agent_history",
    "load_b2c_short_messages",
    "load_b2c_slot_memory",
    "refresh_b2c_short_memory_ttl",
    "save_b2c_slot_memory",
    "sync_b2c_session_memory",
    "sync_b2c_short_memory",
    "touch_b2c_session_redis",
]


def sync_b2c_short_memory(session_id: str, messages: list[Any]) -> int:
    """本场会话消息 → Redis List（全量，按 session_id）。"""
    sid = (session_id or "").strip()
    if not sid:
        return 0
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import sync_b2c_session_conversation_sync

        return int(sync_b2c_session_conversation_sync(sid, messages) or 0)
    except Exception:
        logger.warning("sync short memory failed session_id=%s", sid)
        return 0


def load_b2c_short_messages(session_id: str) -> list[dict[str, str]]:
    """按 session_id 读本场全部短期对话；失败返回空。"""
    sid = (session_id or "").strip()
    if not sid:
        return []
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import context_messages_b2c_session_sync

        return list(context_messages_b2c_session_sync(sid) or [])
    except Exception:
        logger.warning("load short memory failed session_id=%s", sid)
        return []


def _rows_to_langchain_messages(rows: list[dict[str, str]]) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    out: list[Any] = []
    for item in rows:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "user":
            out.append(HumanMessage(content=content))
        elif item.get("role") == "assistant":
            out.append(AIMessage(content=content))
    return out


def load_b2c_agent_history(session_id: str) -> list[Any]:
    """short 全量 + compress 摘要，供进图 Agent 的 history。"""
    sid = (session_id or "").strip()
    rows = load_b2c_short_messages(sid)
    msgs = _rows_to_langchain_messages(rows)
    if not sid or not msgs:
        return msgs
    from algorithm.context_compress import apply_incremental_compress

    return apply_incremental_compress(msgs, session_id=sid)


def save_b2c_slot_memory(
    user_id: str,
    slots: dict[str, list[str]],
    *,
    session_id: str = "",
) -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    ensure_uvgraph_root()
    try:
        from src.redis.slot_memory import save_slot_memory_sync

        save_slot_memory_sync(uid, slots, session_id=session_id)
    except Exception:
        logger.warning("save slot memory failed user_id=%s", uid)


def load_b2c_slot_memory(user_id: str, *, session_id: str = "") -> dict[str, list[str]]:
    uid = (user_id or "").strip()
    if not uid:
        return {"疾病": [], "症状": [], "部位": []}
    ensure_uvgraph_root()
    try:
        from src.redis.slot_memory import load_slot_memory_sync

        return load_slot_memory_sync(uid, session_id=session_id)
    except Exception:
        logger.warning("load slot memory failed user_id=%s", uid)
        return {"疾病": [], "症状": [], "部位": []}


def refresh_b2c_short_memory_ttl(
    user_id: str,
    ttl_seconds: int,
    *,
    session_id: str = "",
) -> None:
    """与 Session 空闲 TTL 对齐，滑动续期 Redis 短期 key。"""
    uid = (user_id or "").strip()
    sid = (session_id or "").strip()
    if (not uid and not sid) or int(ttl_seconds) <= 0:
        return
    ttl = int(ttl_seconds)
    ensure_uvgraph_root()
    try:
        from src.redis.context_compress import compress_key
        from src.redis.conversation import _sync_redis, short_conversation_key
        from src.redis.slot_memory import slot_memory_key

        keys: list[str] = []
        if uid:
            keys.append(slot_memory_key(uid))
        if sid:
            keys.append(short_conversation_key(sid))
            keys.append(compress_key(sid))
        client = _sync_redis()
        try:
            for key in keys:
                if client.exists(key):
                    client.expire(key, ttl)
        finally:
            client.close()
    except Exception:
        logger.warning("refresh short memory ttl failed session_id=%s", sid)


def touch_b2c_session_redis(user_id: str, *, session_id: str = "") -> None:
    from algorithm.session_release import session_idle_seconds

    refresh_b2c_short_memory_ttl(user_id, session_idle_seconds(), session_id=session_id)


def sync_b2c_session_memory(session: Any) -> None:
    """SessionStore 与 Redis 短期对话/槽位镜像，TTL 随 session 空闲策略续期。"""
    uid = str(getattr(session, "group_id", "") or getattr(session, "user_id", "") or "").strip()
    sid = str(getattr(session, "session_id", "") or "").strip()
    if not sid:
        return
    user_rows = list(getattr(session, "user", None) or [])
    sync_b2c_short_memory(sid, user_rows)
    if uid:
        slots = getattr(session, "slot_memory", None) or {}
        save_b2c_slot_memory(uid, slots, session_id=sid)
    touch_b2c_session_redis(uid, session_id=sid)


def clear_b2c_short_memory(
    user_id: str,
    *,
    session_id: str = "",
    clear_slots: bool = True,
) -> None:
    uid = (user_id or "").strip()
    sid = (session_id or "").strip()
    if not sid and not uid:
        return
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import clear_conversation_sync
        from src.redis.context_compress import clear_compress_sync
        from src.redis.slot_memory import clear_slot_memory_sync

        if sid:
            clear_conversation_sync(sid)
            clear_compress_sync(sid)
        if uid and clear_slots:
            clear_slot_memory_sync(uid)
    except Exception:
        logger.warning("clear short memory failed session_id=%s", sid)
