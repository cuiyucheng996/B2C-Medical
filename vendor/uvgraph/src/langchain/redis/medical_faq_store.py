"""医疗 FAQ 运行时存储：启动从 Redis 加载，问诊节点同步读取。"""

from __future__ import annotations

import logging
from typing import Any

from src.langchain.redis.medical_faq_items import MEDICAL_FAQ_ITEMS, match_faq_items, suggest_faq_items
from src.langchain.redis.redis_data_sync import (
    faq_ids_key,
    faq_item_key,
    get_redis_client,
    medical_faq_namespace,
    sync_medical_faq_to_redis,
)
from src.schemas.medical_faq import FaqPhase, MedicalFaqItem

logger = logging.getLogger(__name__)

_CACHE: list[MedicalFaqItem] | None = None


def _hash_to_faq_item(data: dict[str, Any]) -> MedicalFaqItem | None:
    faq_id = str(data.get("id") or "").strip()
    question = str(data.get("question") or "").strip()
    answer = str(data.get("answer") or "").strip()
    if not all([faq_id, question, answer]):
        return None
    try:
        priority = int(str(data.get("priority") or "0"))
    except ValueError:
        priority = 0
    return MedicalFaqItem(
        id=faq_id,
        question=question,
        answer=answer,
        category=data.get("category") or "usage",  # type: ignore[typeddict-item]
        keywords=str(data.get("keywords") or ""),
        phase=data.get("phase") or "any",  # type: ignore[typeddict-item]
        priority=priority,
        action=data.get("action") or "reply_and_continue",  # type: ignore[typeddict-item]
    )


async def fetch_medical_faq_from_redis() -> list[MedicalFaqItem]:
    """从 Redis 读取当前 namespace 下全部医疗 FAQ。"""
    redis_client = await get_redis_client()
    ns = medical_faq_namespace()
    ids = await redis_client.smembers(faq_ids_key(ns))
    items: list[MedicalFaqItem] = []
    for faq_id in sorted(ids):
        raw = await redis_client.hgetall(faq_item_key(ns, faq_id))
        if not raw:
            continue
        item = _hash_to_faq_item(raw)
        if item:
            items.append(item)
    return items


async def get_medical_faq_by_id(faq_id: str) -> MedicalFaqItem | None:
    """按 id 从 Redis 读取单条 FAQ。"""
    text = (faq_id or "").strip()
    if not text:
        return None
    redis_client = await get_redis_client()
    ns = medical_faq_namespace()
    raw = await redis_client.hgetall(faq_item_key(ns, text))
    if not raw:
        return None
    return _hash_to_faq_item(raw)


async def refresh_medical_faq_cache() -> int:
    """从 Redis 刷新进程内缓存，供同步图节点匹配。"""
    global _CACHE
    _CACHE = await fetch_medical_faq_from_redis()
    logger.info("medical faq cache refreshed: %s items from redis", len(_CACHE))
    return len(_CACHE)


def load_medical_faq_cache_from_source() -> int:
    """Redis 不可用时，用 ``medical_faq_items.py`` 兜底。"""
    global _CACHE
    _CACHE = list(MEDICAL_FAQ_ITEMS)
    logger.warning("medical faq cache loaded from source file: %s items", len(_CACHE))
    return len(_CACHE)


def get_medical_faq_cache() -> list[MedicalFaqItem]:
    if _CACHE is None:
        load_medical_faq_cache_from_source()
    return list(_CACHE or [])


def match_medical_faq(
    query: str,
    *,
    phase: FaqPhase | str = "any",
) -> MedicalFaqItem | None:
    """同步匹配 FAQ（读进程内缓存；缓存由启动时 Redis 加载）。"""
    return match_faq_items(get_medical_faq_cache(), query, phase=phase)


def suggest_medical_faq(
    query: str,
    *,
    phase: FaqPhase | str = "any",
    limit: int = 5,
) -> list[MedicalFaqItem]:
    """FAQ 联想（读进程内 Redis 缓存）。"""
    return suggest_faq_items(get_medical_faq_cache(), query, phase=phase, limit=limit)


async def bootstrap_medical_faq(*, replace: bool = True) -> int:
    """启动流程：Python 源数据 → Redis → 进程缓存。"""
    try:
        count, ids_key = await sync_medical_faq_to_redis(replace=replace)
        logger.info("medical faq synced to redis: count=%s ids_key=%s", count, ids_key)
        return await refresh_medical_faq_cache()
    except Exception:
        logger.exception("medical faq redis bootstrap failed; fallback to source file")
        return load_medical_faq_cache_from_source()


__all__ = [
    "bootstrap_medical_faq",
    "fetch_medical_faq_from_redis",
    "get_medical_faq_by_id",
    "get_medical_faq_cache",
    "load_medical_faq_cache_from_source",
    "match_medical_faq",
    "refresh_medical_faq_cache",
    "suggest_medical_faq",
]
