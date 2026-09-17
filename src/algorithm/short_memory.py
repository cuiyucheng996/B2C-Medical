"""B2C 短期对话：Redis key 用 user_id（conversation:short:{user_id}）。"""

from __future__ import annotations

import logging
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "clear_b2c_short_memory",
    "load_b2c_short_messages",
    "load_b2c_slot_memory",
    "save_b2c_slot_memory",
    "sync_b2c_short_memory",
]


def sync_b2c_short_memory(user_id: str, messages: list[Any]) -> int:
    """会话消息 → Redis List，按 user_id 存。"""
    uid = (user_id or "").strip()
    if not uid:
        return 0
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import sync_conversation_from_messages_sync

        return int(sync_conversation_from_messages_sync(uid, messages) or 0)
    except Exception:
        logger.warning("sync short memory failed user_id=%s", uid)
        return 0


def load_b2c_short_messages(user_id: str) -> list[dict[str, str]]:
    """按 user_id 读 Redis 滑动窗口；失败返回空。"""
    uid = (user_id or "").strip()
    if not uid:
        return []
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import context_messages_sync

        return list(context_messages_sync(uid) or [])
    except Exception:
        logger.warning("load short memory failed user_id=%s", uid)
        return []


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


def load_b2c_slot_memory(user_id: str) -> dict[str, list[str]]:
    uid = (user_id or "").strip()
    if not uid:
        return {"疾病": [], "症状": [], "部位": []}
    ensure_uvgraph_root()
    try:
        from src.redis.slot_memory import load_slot_memory_sync

        return load_slot_memory_sync(uid)
    except Exception:
        logger.warning("load slot memory failed user_id=%s", uid)
        return {"疾病": [], "症状": [], "部位": []}


def clear_b2c_short_memory(user_id: str) -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    ensure_uvgraph_root()
    try:
        from src.redis.conversation import clear_conversation_sync
        from src.redis.context_compress import clear_compress_sync
        from src.redis.slot_memory import clear_slot_memory_sync

        clear_conversation_sync(uid)
        clear_slot_memory_sync(uid)
        clear_compress_sync(uid)
    except Exception:
        logger.warning("clear short memory failed user_id=%s", uid)
