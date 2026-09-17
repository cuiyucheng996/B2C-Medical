"""检索结果缓存（进程内，评测/陪聊共用）。"""

from __future__ import annotations

import hashlib
import time
from typing import Any

__all__ = ["get_cached_hits", "set_cached_hits"]

_STORE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _key(channel: str, query: str, handler: str) -> str:
    raw = f"{channel}|{handler}|{query}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_cached_hits(channel: str, query: str, handler: str, *, ttl: int) -> list[dict[str, Any]] | None:
    if ttl <= 0:
        return None
    key = _key(channel, query, handler)
    row = _STORE.get(key)
    if not row:
        return None
    expires, hits = row
    if time.time() > expires:
        _STORE.pop(key, None)
        return None
    return list(hits)


def set_cached_hits(
    channel: str,
    query: str,
    handler: str,
    hits: list[dict[str, Any]],
    *,
    ttl: int,
) -> None:
    if ttl <= 0:
        return
    key = _key(channel, query, handler)
    _STORE[key] = (time.time() + ttl, list(hits))
