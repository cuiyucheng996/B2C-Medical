"""会话槽位（部位/病/症）：按 user_id 存 JSON，会话结束随短期对话一起清。"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.configs import REDIS_CONFIG
from src.redis.conversation import SHORT_TTL_SECONDS, _sync_redis

logger = logging.getLogger(__name__)

SLOT_NS = str(REDIS_CONFIG.get("short_slot_namespace") or "conversation:slots").strip()
_SLOT_KEYS = ("疾病", "症状", "部位")

__all__ = ["clear_slot_memory_sync", "load_slot_memory_sync", "save_slot_memory_sync", "slot_memory_key"]


def slot_memory_key(user_id: str) -> str:
    return f"{SLOT_NS}:{(user_id or '').strip()}"


def save_slot_memory_sync(user_id: str, slots: dict[str, list[str]], *, session_id: str = "") -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    payload = {
        "session_id": (session_id or "").strip(),
        "slots": {key: list(slots.get(key) or []) for key in _SLOT_KEYS},
    }
    client = _sync_redis()
    try:
        key = slot_memory_key(uid)
        client.set(key, json.dumps(payload, ensure_ascii=False))
        if SHORT_TTL_SECONDS > 0:
            client.expire(key, SHORT_TTL_SECONDS)
    finally:
        client.close()


def load_slot_memory_sync(user_id: str, *, session_id: str = "") -> dict[str, list[str]]:
    uid = (user_id or "").strip()
    empty = {key: [] for key in _SLOT_KEYS}
    if not uid:
        return empty
    client = _sync_redis()
    try:
        raw = client.get(slot_memory_key(uid))
    finally:
        client.close()
    if not raw:
        return empty
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return empty
    want = (session_id or "").strip()
    stored_sid = str(data.get("session_id") or "").strip() if isinstance(data, dict) else ""
    if want and stored_sid and stored_sid != want:
        return empty
    slots = data.get("slots") if isinstance(data, dict) else None
    if not isinstance(slots, dict):
        return empty
    out = {key: [] for key in _SLOT_KEYS}
    for key in _SLOT_KEYS:
        out[key] = [str(item).strip() for item in (slots.get(key) or []) if str(item).strip()]
    return out


def clear_slot_memory_sync(user_id: str) -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    client = _sync_redis()
    try:
        client.delete(slot_memory_key(uid))
    finally:
        client.close()
