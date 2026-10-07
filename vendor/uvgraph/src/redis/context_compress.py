"""增量上下文压缩：已压片段存 Redis（键=session_id），只压新落入窗口外的对话。"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.configs import REDIS_CONFIG
from src.redis.conversation import SHORT_TTL_SECONDS, _sync_redis

logger = logging.getLogger(__name__)

COMPRESS_NS = str(REDIS_CONFIG.get("short_compress_namespace") or "conversation:compress").strip()

__all__ = [
    "clear_compress_sync",
    "compress_key",
    "load_compress_sync",
    "save_compress_sync",
]


def compress_key(session_id: str) -> str:
    return f"{COMPRESS_NS}:{(session_id or '').strip()}"


def load_compress_sync(session_id: str) -> dict[str, Any]:
    sid = (session_id or "").strip()
    empty = {"covered_humans": 0, "segments": []}
    if not sid:
        return empty
    client = _sync_redis()
    try:
        raw = client.get(compress_key(sid))
    finally:
        client.close()
    if not raw:
        return empty
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    segs = data.get("segments") if isinstance(data.get("segments"), list) else []
    return {
        "covered_humans": int(data.get("covered_humans") or 0),
        "segments": [str(item).strip() for item in segs if str(item).strip()],
    }


def save_compress_sync(session_id: str, payload: dict[str, Any]) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    client = _sync_redis()
    try:
        key = compress_key(sid)
        client.set(key, json.dumps(payload, ensure_ascii=False))
        if SHORT_TTL_SECONDS > 0:
            client.expire(key, SHORT_TTL_SECONDS)
    finally:
        client.close()


def clear_compress_sync(session_id: str) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    client = _sync_redis()
    try:
        client.delete(compress_key(sid))
    finally:
        client.close()
