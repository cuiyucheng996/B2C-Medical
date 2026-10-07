"""多问法粗召回后的精排（RRF）；当前未接入 Agent。"""

from .algorithm import (
    ContextDocument,
    DEFAULT_RRF_K,
    DocumentDedupStore,
    RerankedDocument,
    dedupe_documents,
    document_content_hash,
    document_to_utf8,
    merge_retrieval_documents,
    normalize_document_text,
    rrf_rerank,
)

__all__ = [
    "ContextDocument",
    "DEFAULT_RRF_K",
    "DocumentDedupStore",
    "RerankedDocument",
    "dedupe_documents",
    "document_content_hash",
    "document_to_utf8",
    "merge_retrieval_documents",
    "normalize_document_text",
    "rrf_rerank",
]
