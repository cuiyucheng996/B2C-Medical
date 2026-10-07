"""疾病知识库 Chroma 存储与检索。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.configs.cache import CHROMA_CACHE_DIR
from src.embeddings import get_embeddings
from src.knowledge.parser import DEFAULT_DISEASE_MD, DiseaseRecord, parse_disease_markdown

logger = logging.getLogger(__name__)

CHROMA_DIR = CHROMA_CACHE_DIR
COLLECTION_NAME = "disease_kb"
KB_VERSION = "v1"


def get_disease_vectorstore() -> Chroma:
    """打开已持久化的疾病向量库（嵌入式 Chroma，无需单独起服务）。"""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _records_to_documents(records: list[DiseaseRecord]) -> list[Document]:
    docs: list[Document] = []
    for record in records:
        metadata = record.to_metadata()
        metadata["kb_version"] = KB_VERSION
        docs.append(
            Document(
                page_content=record.to_index_text(),
                metadata=metadata,
                id=record.disease_name,
            )
        )
    return docs


def ingest_disease_kb(
    *,
    source_path: Path | None = None,
    reset: bool = True,
) -> dict[str, Any]:
    """离线入库：解析 disease.md → embedding → 写入本地 Chroma。"""
    records = parse_disease_markdown(source_path)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        if any(col.name == COLLECTION_NAME for col in client.list_collections()):
            client.delete_collection(COLLECTION_NAME)
            logger.info("deleted existing collection %s", COLLECTION_NAME)

    documents = _records_to_documents(records)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        ids=[doc.id for doc in documents],
    )

    return {
        "collection": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
        "count": len(records),
        "source": str(source_path or DEFAULT_DISEASE_MD),
        "kb_version": KB_VERSION,
        "vectorstore": vectorstore,
    }


def search_diseases(query: str, *, k: int = 5) -> list[dict[str, Any]]:
    """按用户描述检索候选疾病。"""
    text = query.strip()
    if not text:
        return []

    store = get_disease_vectorstore()
    hits = store.similarity_search_with_score(text, k=k)
    results: list[dict[str, Any]] = []
    for doc, score in hits:
        results.append(
            {
                "disease_name": doc.metadata.get("disease_name", ""),
                "symptom_category": doc.metadata.get("symptom_category", ""),
                "symptoms": doc.metadata.get("symptoms", ""),
                "synonyms": doc.metadata.get("synonyms", ""),
                "content": doc.page_content,
                "score": float(score),
            }
        )
    return results
