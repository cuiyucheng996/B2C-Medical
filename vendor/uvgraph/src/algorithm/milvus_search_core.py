"""Milvus 疾病向量检索（仅应在子进程内 import pymilvus）。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from src.configs import MILVUS_CONFIG
from src.configs.proxy import ensure_local_no_proxy
from src.embeddings import get_cpu_embeddings
from src.knowledge.body_disease_map import diseases_for_body_part

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_EMBED_WORKER = _ROOT / "scripts" / "embed_texts_worker.py"

DEFAULT_DISEASE_COLLECTION = "disease"
DEFAULT_MAX_DISTANCE = 0.65
DEFAULT_TOP_K = 5

DEFAULT_ENTITY_MAX_DISTANCE = 0.5

__all__ = [
    "search_diseases_in_milvus_core",
    "search_entity_in_milvus_core",
]


def _milvus_client_kwargs() -> dict[str, Any]:
    kwargs = dict(MILVUS_CONFIG)
    if not str(kwargs.get("token") or "").strip():
        kwargs.pop("token", None)
    # RPC 超时：Milvus 不可用時更快失败；服务正常时检索逻辑与结果不变
    raw_timeout = kwargs.pop("timeout", None)
    if raw_timeout is None:
        raw_timeout = kwargs.pop("rpc_timeout_sec", None)
    if raw_timeout is not None:
        kwargs["timeout"] = float(raw_timeout)
    return kwargs


@lru_cache
def _get_milvus_client() -> MilvusClient:
    ensure_local_no_proxy()
    return MilvusClient(**_milvus_client_kwargs())


def _disease_collection_name() -> str:
    name = str(MILVUS_CONFIG.get("disease_collection") or DEFAULT_DISEASE_COLLECTION).strip()
    return name or DEFAULT_DISEASE_COLLECTION


def _embed_query_isolated(text: str) -> list[float]:
    """子进程 embedding，避免 pymilvus 与 transformers 同进程冲突。"""
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "texts.json"
        out_path = Path(tmp) / "vectors.json"
        in_path.write_text(json.dumps([text], ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(_EMBED_WORKER), str(in_path), str(out_path)],
            check=True,
            cwd=str(_ROOT),
        )
        vectors = json.loads(out_path.read_text(encoding="utf-8"))
    return list(vectors[0])


def _parse_metadata_text(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def _normalize_disease_hit(item: dict[str, Any]) -> dict[str, Any]:
    entity = item.get("entity") or {}
    text = str(entity.get("text") or "")
    meta = _parse_metadata_text(text)
    disease = str(entity.get("disease") or meta.get("疾病") or "").strip()
    distance = item.get("distance")
    return {
        "disease": disease,
        "score": float(distance) if distance is not None else None,
        "symptom_category": meta.get("症状类型", ""),
        "symptoms": meta.get("包含症状", ""),
        "synonyms": meta.get("通俗同义词", ""),
        "detail_text": text,
    }


def search_diseases_in_milvus_core(
    query: str,
    *,
    body_part: str | None = None,
    limit: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Milvus 稠密向量 RAG：query embed → COSINE 检索 → 结构化命中列表。"""
    text = (query or "").strip()
    if not text:
        return []

    collection = collection_name or _disease_collection_name()
    client = _get_milvus_client()
    if not client.has_collection(collection):
        logger.warning("milvus collection not found: %s", collection)
        return []

    query_vector = get_cpu_embeddings().embed_query(text)
    allowed = set(diseases_for_body_part(body_part)) if body_part else None
    search_limit = max(limit * 10, 50) if allowed is not None else max(1, min(limit, 10))
    hits = client.search(
        collection_name=collection,
        data=[query_vector],
        anns_field="vector",
        output_fields=["disease", "text"],
        limit=search_limit,
    )

    raw_items = hits[0] if hits else []
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
    pool.sort(key=lambda item: item.get("score") if item.get("score") is not None else 1.0)
    for normalized in pool:
        disease = normalized["disease"]
        if disease in seen:
            continue
        seen.add(disease)
        merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _normalize_entity_hit(item: dict[str, Any]) -> dict[str, Any]:
    entity = item.get("entity") or {}
    text = str(entity.get("text") or "")
    meta = _parse_metadata_text(text)
    name = str(entity.get("disease") or meta.get("部位") or meta.get("症状") or meta.get("证型") or "").strip()
    distance = item.get("distance")
    node_id = meta.get("mysql_id")
    return {
        "standard_name": name,
        "distance": float(distance) if distance is not None else None,
        "node_id": str(node_id) if node_id else None,
        "detail_text": text,
    }


def search_entity_in_milvus_core(
    query: str,
    *,
    collection_names: list[str],
    allowed_names: set[str] | None = None,
    limit: int = 3,
    max_distance: float = DEFAULT_ENTITY_MAX_DISTANCE,
) -> list[dict[str, Any]]:
    """标准名向量检索：query embed → Milvus COSINE → 结构化命中（含 mysql_id）。"""
    text = (query or "").strip()
    if not text or not collection_names:
        return []

    client = _get_milvus_client()
    # 主进程顺序：UIE 之后 CPU 编码，再请求 Docker Milvus
    query_vector = get_cpu_embeddings().embed_query(text)
    search_limit = max(limit * 10, 30) if allowed_names else max(1, min(limit, 10))
    merged: list[dict[str, Any]] = []

    for collection_name in collection_names:
        if not client.has_collection(collection_name):
            logger.warning("milvus collection not found: %s", collection_name)
            continue
        hits = client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            output_fields=["disease", "text"],
            limit=search_limit,
        )
        for item in hits[0] if hits else []:
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

    # COSINE 返回值越大越像，去重后按相似度从高到低截断
    merged.sort(key=lambda row: row.get("distance") if row.get("distance") is not None else -1.0, reverse=True)
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
