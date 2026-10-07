"""向量检索门面：vector_store.backend=chroma|milvus。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "run_milvus_entity_search_isolated",
    "run_milvus_search_isolated",
    "shutdown_milvus_executor",
    "vector_store_backend",
]


def vector_store_backend() -> str:
    from src.configs.config_loader import VECTOR_STORE_BACKEND

    return (VECTOR_STORE_BACKEND or "milvus").strip().lower()


def shutdown_milvus_executor() -> None:
    return


def run_milvus_search_isolated(
    query: str,
    *,
    body_part: str | None = None,
    limit: int = 5,
    max_distance: float = 0.65,
    collection_name: str | None = None,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """主进程内检索疾病向量；timeout 保留兼容，不再开子进程。"""
    del timeout
    text = (query or "").strip()
    if not text:
        return []
    try:
        if vector_store_backend() == "chroma":
            from src.algorithm.chroma_search_core import search_diseases_in_chroma_core

            result = search_diseases_in_chroma_core(
                text,
                body_part=body_part,
                limit=limit,
                max_distance=max_distance,
                collection_name=collection_name,
            )
            return list(result) if isinstance(result, list) else []
        from src.algorithm.milvus_search_core import search_diseases_in_milvus_core

        result = search_diseases_in_milvus_core(
            text,
            body_part=body_part,
            limit=limit,
            max_distance=max_distance,
            collection_name=collection_name,
        )
        return list(result) if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("milvus search failed query=%r: %s", text[:80], exc)
        return []


def run_milvus_entity_search_isolated(
    query: str,
    *,
    collection_names: list[str],
    allowed_names: set[str] | None = None,
    limit: int = 3,
    max_distance: float = 0.5,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """主进程内：CPU 编码 mention → Docker Milvus。"""
    del timeout
    text = (query or "").strip()
    if not text or not collection_names:
        return []
    try:
        if vector_store_backend() == "chroma":
            from src.algorithm.chroma_search_core import search_entity_in_chroma_core

            result = search_entity_in_chroma_core(
                text,
                collection_names=list(collection_names),
                allowed_names=set(allowed_names) if allowed_names else None,
                limit=limit,
                max_distance=max_distance,
            )
            return list(result) if isinstance(result, list) else []
        from src.algorithm.milvus_search_core import search_entity_in_milvus_core

        result = search_entity_in_milvus_core(
            text,
            collection_names=list(collection_names),
            allowed_names=set(allowed_names) if allowed_names else None,
            limit=limit,
            max_distance=max_distance,
        )
        return list(result) if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("milvus entity search failed query=%r: %s", text[:80], exc)
        return []
