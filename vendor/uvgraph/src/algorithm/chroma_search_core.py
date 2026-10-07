"""Chroma 向量检索（与 milvus_search_core 返回结构对齐）。"""

from __future__ import annotations

import logging
from typing import Any

from src.algorithm.milvus_search_core import (
    DEFAULT_DISEASE_COLLECTION,
    DEFAULT_ENTITY_MAX_DISTANCE,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_TOP_K,
    _normalize_disease_hit,
    _normalize_entity_hit,
    _parse_metadata_text,
)
from src.embeddings import get_cpu_embeddings
from src.knowledge.body_disease_map import diseases_for_body_part
from src.knowledge.chroma_store import get_chroma_collection

logger = logging.getLogger(__name__)

__all__ = [
    "search_diseases_in_chroma_core",
    "search_entity_in_chroma_core",
]


def _chroma_distance_to_score(distance: float | None) -> float | None:
    if distance is None:
        return None
    # Chroma cosine distance：越小越相似；与现有 Milvus 过滤逻辑对齐为「分数」
    return float(distance)


def search_diseases_in_chroma_core(
    query: str,
    *,
    body_part: str | None = None,
    limit: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    text = (query or "").strip()
    if not text:
        return []
    collection = collection_name or DEFAULT_DISEASE_COLLECTION
    try:
        col = get_chroma_collection(collection, create=False)
    except Exception:
        logger.warning("chroma collection not found: %s", collection)
        return []

    query_vector = get_cpu_embeddings().embed_query(text)
    allowed = set(diseases_for_body_part(body_part)) if body_part else None
    search_limit = max(limit * 10, 50) if allowed is not None else max(1, min(limit, 10))
    hits = col.query(
        query_embeddings=[query_vector],
        n_results=search_limit,
        include=["metadatas", "documents", "distances"],
    )
    documents = (hits.get("documents") or [[]])[0]
    metadatas = (hits.get("metadatas") or [[]])[0]
    distances = (hits.get("distances") or [[]])[0]
    raw_items: list[dict[str, Any]] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        meta = meta if isinstance(meta, dict) else {}
        disease = str(meta.get("disease") or "").strip()
        raw_items.append(
            {
                "entity": {"disease": disease, "text": str(doc or "")},
                "distance": _chroma_distance_to_score(dist),
            }
        )

    if allowed is None:
        results: list[dict[str, Any]] = []
        for item in raw_items:
            normalized = _normalize_disease_hit(item)
            if not normalized["disease"]:
                continue
            score = normalized["score"]
            if score is not None and score > max_distance:
                continue
            results.append(normalized)
        return results[:limit]

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in raw_items[:15]:
        normalized = _normalize_disease_hit(item)
        disease = normalized["disease"]
        if not disease or disease not in allowed or disease in seen:
            continue
        seen.add(disease)
        merged.append(normalized)
        if len(merged) >= limit:
            return merged
    pool: list[dict[str, Any]] = []
    for item in raw_items:
        normalized = _normalize_disease_hit(item)
        disease = normalized["disease"]
        if not disease or disease not in allowed:
            continue
        pool.append(normalized)
    pool.sort(key=lambda row: row.get("score") if row.get("score") is not None else 1.0)
    for normalized in pool:
        disease = normalized["disease"]
        if disease in seen:
            continue
        seen.add(disease)
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def search_entity_in_chroma_core(
    query: str,
    *,
    collection_names: list[str],
    allowed_names: set[str] | None = None,
    limit: int = 3,
    max_distance: float = DEFAULT_ENTITY_MAX_DISTANCE,
) -> list[dict[str, Any]]:
    text = (query or "").strip()
    if not text or not collection_names:
        return []
    query_vector = get_cpu_embeddings().embed_query(text)
    search_limit = max(limit * 10, 30) if allowed_names else max(1, min(limit, 10))
    merged: list[dict[str, Any]] = []

    for collection_name in collection_names:
        try:
            col = get_chroma_collection(collection_name, create=False)
        except Exception:
            logger.warning("chroma collection not found: %s", collection_name)
            continue
        hits = col.query(
            query_embeddings=[query_vector],
            n_results=search_limit,
            include=["metadatas", "documents", "distances"],
        )
        documents = (hits.get("documents") or [[]])[0]
        metadatas = (hits.get("metadatas") or [[]])[0]
        distances = (hits.get("distances") or [[]])[0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta = meta if isinstance(meta, dict) else _parse_metadata_text(str(doc or ""))
            item = {
                "entity": {
                    "disease": str(meta.get("disease") or ""),
                    "text": str(doc or ""),
                },
                "distance": _chroma_distance_to_score(dist),
            }
            normalized = _normalize_entity_hit(item)
            name = normalized["standard_name"]
            if not name:
                continue
            distance = normalized["distance"]
            if max_distance < 0.99 and distance is not None and distance > max_distance:
                continue
            if allowed_names is not None and name not in allowed_names:
                continue
            merged.append(normalized)

    merged.sort(key=lambda row: row.get("distance") if row.get("distance") is not None else 999.0)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in merged:
        name = str(row.get("standard_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique
