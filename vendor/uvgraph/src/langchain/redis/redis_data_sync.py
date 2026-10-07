"""FAQ 数据同步到 Redis（异步）。

医疗问诊 FAQ：``medical_faq_items.MEDICAL_FAQ_ITEMS`` + namespace ``medical_faq``
"""

from __future__ import annotations

import logging
from typing import Any

from redis.asyncio import Redis

from src.configs import REDIS_CONFIG

logger = logging.getLogger(__name__)

REDIS_CLIENT: Redis | None = None

_FAQ_HASH_FIELDS = (
    "id",
    "question",
    "answer",
    "category",
    "keywords",
    "phase",
    "priority",
    "action",
)

__all__ = [
    "faq_ids_key",
    "faq_item_key",
    "faq_meta_key",
    "get_redis_client",
    "medical_faq_namespace",
    "sync_faq_to_redis",
    "sync_medical_faq_to_redis",
]


async def get_redis_client() -> Redis:
    global REDIS_CLIENT
    if REDIS_CLIENT is None:
        REDIS_CLIENT = Redis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            db=REDIS_CONFIG["db"],
            password=REDIS_CONFIG.get("password") or None,
            decode_responses=True,
        )
    return REDIS_CLIENT


def medical_faq_namespace() -> str:
    return str(REDIS_CONFIG.get("medical_faq_redis_namespace") or "medical_faq").strip()


def faq_item_key(namespace: str, faq_id: str) -> str:
    return f"{namespace}:item:{faq_id}"


def faq_ids_key(namespace: str) -> str:
    return f"{namespace}:ids"


def faq_meta_key(namespace: str) -> str:
    return f"{namespace}:meta"


def _faq_item_mapping(item: dict[str, Any]) -> dict[str, str]:
    """将 FAQ 条目转为 Redis Hash 字段（字符串值）。"""
    faq_id = str(item.get("id") or "").strip()
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    mapping: dict[str, str] = {
        "id": faq_id,
        "question": question,
        "answer": answer,
    }
    for field in _FAQ_HASH_FIELDS:
        if field in ("id", "question", "answer"):
            continue
        if field in item and item[field] is not None:
            mapping[field] = str(item[field]).strip()
    if "category" not in mapping:
        mapping["category"] = "usage"
    if "keywords" not in mapping:
        mapping["keywords"] = ""
    if "phase" not in mapping:
        mapping["phase"] = "any"
    if "priority" not in mapping:
        mapping["priority"] = "0"
    if "action" not in mapping:
        mapping["action"] = "reply_and_continue"
    return mapping


async def sync_faq_to_redis(
    *,
    namespace: str,
    items: list[dict[str, str]],
    replace: bool = False,
) -> tuple[int, str]:
    """将 FAQ 批量写入 Redis。

    键约定：
    - ``{namespace}:item:{id}`` — Hash（id / question / answer / …）
    - ``{namespace}:ids``     — Set，全部 FAQ id
    - ``{namespace}:meta``    — Hash，``count`` / ``kind`` 字段

    ``replace=True`` 时先清空该 namespace 下已有键。
    返回 ``(写入条数, ids_key)``。
    """
    redis_client = await get_redis_client()
    ns = namespace.strip()
    if not ns:
        raise ValueError("namespace is required")
    if not items:
        raise ValueError("items is required")

    ids_key = faq_ids_key(ns)
    meta_key = faq_meta_key(ns)

    if replace:
        keys_to_delete = [key async for key in redis_client.scan_iter(match=f"{ns}:*")]
        if keys_to_delete:
            async with redis_client.pipeline(transaction=True) as pipe:
                for key in keys_to_delete:
                    pipe.delete(key)
                await pipe.execute()

    async with redis_client.pipeline(transaction=True) as pipe:
        written = 0
        for item in items:
            mapping = _faq_item_mapping(item)
            faq_id = mapping["id"]
            question = mapping["question"]
            answer = mapping["answer"]
            if not all([faq_id, question, answer]):
                logger.warning("skip invalid faq item: %r", item)
                continue

            pipe.hset(faq_item_key(ns, faq_id), mapping=mapping)
            pipe.sadd(ids_key, faq_id)
            written += 1

        pipe.hset(meta_key, mapping={"count": str(written), "kind": ns})
        await pipe.execute()

    logger.info("synced %s faq items to redis namespace=%s replace=%s", written, ns, replace)
    return written, ids_key


async def sync_medical_faq_to_redis(*, replace: bool = True) -> tuple[int, str]:
    """将 ``medical_faq_items.MEDICAL_FAQ_ITEMS`` 同步到医疗 FAQ namespace。"""
    from src.langchain.redis.medical_faq_items import MEDICAL_FAQ_ITEMS

    return await sync_faq_to_redis(
        namespace=medical_faq_namespace(),
        items=list(MEDICAL_FAQ_ITEMS),
        replace=replace,
    )


if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        count, ids_key = await sync_medical_faq_to_redis(replace=True)
        print(f"synced {count} medical faq items, ids_key={ids_key}")

    asyncio.run(_main())
