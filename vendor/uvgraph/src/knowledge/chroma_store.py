"""Chroma：Docker HttpClient 或本地 PersistentClient。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "chroma_client_mode",
    "chroma_persist_directory",
    "delete_chroma_collection",
    "get_chroma_client",
    "get_chroma_collection",
    "insert_chroma_rows",
    "list_chroma_collections",
    "query_chroma_all",
]


def chroma_client_mode() -> str:
    from src.configs.config_loader import CHROMA_VECTOR_CONFIG

    host = str(CHROMA_VECTOR_CONFIG.get("server_host") or "").strip()
    return "http" if host else "embedded"


def chroma_persist_directory() -> Path:
    from src.configs.config_loader import CHROMA_VECTOR_CONFIG

    from src.configs.cache import CHROMA_CACHE_DIR

    raw = str(CHROMA_VECTOR_CONFIG.get("persist_directory") or CHROMA_CACHE_DIR).strip()
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


@lru_cache
def get_chroma_client() -> Any:
    import chromadb

    from src.configs.config_loader import CHROMA_VECTOR_CONFIG

    host = str(CHROMA_VECTOR_CONFIG.get("server_host") or "").strip()
    if host:
        port = int(CHROMA_VECTOR_CONFIG.get("server_port") or 8000)
        ssl = bool(CHROMA_VECTOR_CONFIG.get("server_ssl"))
        timeout = float(CHROMA_VECTOR_CONFIG.get("connect_timeout_sec") or 5.0)
        _ensure_chroma_http_reachable(host, port, timeout=timeout)
        logger.info("chroma HttpClient %s:%s", host, port)
        return chromadb.HttpClient(host=host, port=port, ssl=ssl)
    path = chroma_persist_directory()
    logger.info("chroma PersistentClient path=%s", path)
    return chromadb.PersistentClient(path=str(path))


def _ensure_chroma_http_reachable(host: str, port: int, *, timeout: float) -> None:
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except OSError as exc:
        persist = chroma_persist_directory()
        raise ConnectionError(
            f"Chroma HTTP {host}:{port} 不可达：{exc}。"
            f"请先执行 docker compose -f deploy/chroma/docker-compose.yml up -d；"
            f"或清空 config.yaml 的 chroma.server_host，改用嵌入式目录 {persist}"
        ) from exc


def list_chroma_collections() -> list[str]:
    client = get_chroma_client()
    try:
        cols = client.list_collections()
        return [str(getattr(c, "name", c)) for c in cols]
    except Exception:
        return []


def get_chroma_collection(name: str, *, create: bool = True) -> Any:
    client = get_chroma_client()
    if create:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=name)


def delete_chroma_collection(name: str) -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass


def _sanitize_meta(row: dict[str, Any]) -> dict[str, Any]:
    skip = {"text", "document", "vector", "vector_text", "embedding", "embeddings"}
    meta: dict[str, Any] = {}
    for key, value in row.items():
        if key in skip or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = str(value)[:512]
    if "disease" not in meta and row.get("name"):
        meta["disease"] = str(row.get("name"))[:256]
    return meta


def insert_chroma_rows(
    collection_name: str,
    rows: list[dict[str, Any]],
    *,
    reset: bool = False,
    id_prefix: str = "",
    batch_size: int = 64,
) -> int:
    if reset:
        delete_chroma_collection(collection_name)
    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        doc = str(row.get("text") or row.get("document") or "").strip()
        if not doc:
            continue
        prepared.append(
            {
                "id": str(row.get("id") or f"{id_prefix or collection_name}-{idx}"),
                "text": doc,
                "vector_text": str(row.get("vector_text") or row.get("disease") or row.get("name") or doc),
                "meta": _sanitize_meta(row),
            }
        )
    if not prepared:
        return 0
    from src.embeddings import get_embeddings

    col = get_chroma_collection(collection_name, create=True)
    inserted = 0
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        vectors = get_embeddings().embed_documents([item["vector_text"] for item in chunk])
        col.add(
            ids=[item["id"] for item in chunk],
            documents=[item["text"] for item in chunk],
            embeddings=list(vectors),
            metadatas=[item["meta"] for item in chunk],
        )
        inserted += len(chunk)
    logger.info("chroma inserted %s rows into %s", inserted, collection_name)
    return inserted


def query_chroma_all(collection_name: str, *, limit: int = 50000) -> list[dict[str, Any]]:
    try:
        col = get_chroma_collection(collection_name, create=False)
    except Exception:
        return []
    try:
        data = col.get(limit=limit, include=["metadatas", "documents"])
    except Exception:
        return []
    docs = list(data.get("documents") or [])
    metas = list(data.get("metadatas") or [])
    rows: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        meta = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        rows.append(
            {
                "disease": str(meta.get("disease") or ""),
                "text": str(doc or ""),
            }
        )
    return rows
