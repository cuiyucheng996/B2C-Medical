"""相似问法语义缓存：FAQ 短路之后、进意图/主图之前可命中整段回复。"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "lookup_semantic_reply",
    "remember_semantic_reply",
    "semantic_cache_enabled",
    "should_try_semantic_cache",
]

_REDIS_NS = "companion:semantic"
_MAX_QUERY_LEN = 120
_PROCESS_FALLBACK: dict[str, list[dict[str, Any]]] = {}


def semantic_cache_enabled() -> bool:
    raw = (os.environ.get("COMPANION_SEMANTIC_CACHE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _threshold() -> float:
    try:
        return float(os.environ.get("COMPANION_SEMANTIC_THRESHOLD") or "0.88")
    except ValueError:
        return 0.88


def _ttl_seconds() -> int:
    try:
        return max(60, int(os.environ.get("COMPANION_SEMANTIC_TTL_SECONDS") or "86400"))
    except ValueError:
        return 86400


def _max_entries() -> int:
    try:
        return max(5, int(os.environ.get("COMPANION_SEMANTIC_MAX_ENTRIES") or "40"))
    except ValueError:
        return 40


def _history_sig(history: list[Any]) -> str:
    n = len(history or [])
    if n == 0:
        return "0"
    last = history[-1]
    tail = str(getattr(last, "content", "") or "")[:96]
    return f"{n}:{hash(tail)}"


def should_try_semantic_cache(
    text: str,
    *,
    preprocess: dict[str, Any] | None = None,
    hitl_resume: bool = False,
) -> bool:
    if not semantic_cache_enabled() or hitl_resume:
        return False
    q = (text or "").strip()
    if not q or len(q) > _MAX_QUERY_LEN:
        return False
    prep = preprocess or {}
    if prep.get("attachments") or prep.get("uploads") or prep.get("has_attachment"):
        return False
    if prep.get("intent_force_full"):
        return False
    return True


def _redis_key(user_id: str) -> str:
    return f"{_REDIS_NS}:{(user_id or '').strip() or 'anon'}"


def _sync_redis() -> Any | None:
    try:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.redis.conversation import _sync_redis

        return _sync_redis()
    except Exception:
        logger.debug("semantic cache redis unavailable", exc_info=True)
        return None


def _load_entries(user_id: str) -> list[dict[str, Any]]:
    client = _sync_redis()
    if client is not None:
        try:
            raw = client.get(_redis_key(user_id))
            if raw:
                data = json.loads(raw)
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
        except Exception:
            logger.debug("semantic cache load failed", exc_info=True)
    return list(_PROCESS_FALLBACK.get(user_id) or [])


def _save_entries(user_id: str, entries: list[dict[str, Any]]) -> None:
    client = _sync_redis()
    payload = json.dumps(entries, ensure_ascii=False)
    if client is not None:
        try:
            client.set(_redis_key(user_id), payload, ex=_ttl_seconds())
            return
        except Exception:
            logger.debug("semantic cache save failed", exc_info=True)
    _PROCESS_FALLBACK[user_id] = entries[: _max_entries()]


def _embed_query(text: str) -> list[float] | None:
    try:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.embeddings import get_embeddings

        emb = get_embeddings()
        vec = emb.embed_query((text or "").strip())
        if not vec:
            return None
        return [float(x) for x in vec]
    except Exception:
        logger.debug("semantic cache embed failed", exc_info=True)
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (na * nb)


def lookup_semantic_reply(
    user_id: str,
    text: str,
    history: list[Any],
) -> tuple[str, dict[str, Any]] | None:
    """命中则返回 (reply, meta_fragment)。"""
    if not should_try_semantic_cache(text):
        return None
    vec = _embed_query(text)
    if vec is None:
        return None
    sig = _history_sig(history)
    now = time.time()
    ttl = _ttl_seconds()
    best: tuple[float, dict[str, Any]] | None = None
    for row in _load_entries(user_id):
        if str(row.get("history_sig") or "") != sig:
            continue
        ts = float(row.get("ts") or 0)
        if ts and now - ts > ttl:
            continue
        row_vec = row.get("v")
        if not isinstance(row_vec, list):
            continue
        sim = _cosine(vec, [float(x) for x in row_vec])
        if sim >= _threshold() and (best is None or sim > best[0]):
            best = (sim, row)
    if best is None:
        return None
    sim, row = best
    reply = str(row.get("reply") or "").strip()
    if not reply:
        return None
    return reply, {
        "semantic_cache": True,
        "semantic_similarity": round(sim, 4),
        "semantic_matched_query": str(row.get("q") or ""),
        "skipped_long_memory": True,
        "bound_tools": [],
        "intent_tasks": [],
    }


def remember_semantic_reply(
    user_id: str,
    text: str,
    reply: str,
    history: list[Any],
    *,
    source: str = "",
) -> None:
    if not semantic_cache_enabled():
        return
    q = (text or "").strip()
    r = (reply or "").strip()
    if not q or not r or len(q) > _MAX_QUERY_LEN:
        return
    if len(r) > 800:
        return
    src = (source or "").strip()
    if src in ("graph", "tool", "hitl") or src.startswith("intent:"):
        return
    vec = _embed_query(q)
    if vec is None:
        return
    sig = _history_sig(history)
    row = {
        "q": q,
        "reply": r,
        "v": vec,
        "history_sig": sig,
        "ts": time.time(),
        "source": src or "turn",
    }
    entries = [e for e in _load_entries(user_id) if str(e.get("q") or "") != q]
    entries.append(row)
    entries.sort(key=lambda x: float(x.get("ts") or 0), reverse=True)
    _save_entries(user_id, entries[: _max_entries()])
