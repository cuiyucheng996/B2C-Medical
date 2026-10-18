"""报告解读用的手册和向量检索。"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["lookup_report_handbook", "search_chroma_knowledge", "search_exam_handbook"]


def search_exam_handbook(
    *,
    query: str,
    mentions: dict[str, list[str]] | None = None,
    alignments: list[dict[str, Any]] | None = None,
    handler: str = "",
) -> dict[str, Any]:
    """只查 RAGFlow / 本地检查手册分块。"""
    from algorithm.kg_query import (
        _aligned_names,
        _pack,
        _prefer_aligned_items,
        _resolve_seeds,
        _summarize,
    )
    from algorithm.ragflow_kb import retrieve_chunks

    name = (handler or "").strip() or "kg_exam_meaning"
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import run_rag_pipeline

        if is_rag_enhanced():
            pack = run_rag_pipeline(
                "handbook",
                query=query,
                handler=name,
                mentions=mentions,
                alignments=alignments,
            )
            names = _aligned_names(list(alignments or []))
            pack["items"] = _prefer_aligned_items(list(pack.get("items") or []), names)
            pack["summary"] = _summarize(name, list(pack.get("seeds") or names), pack["items"])
            return pack
    except Exception:
        logger.warning("rag handbook pipeline failed", exc_info=True)
    used_alignments = list(alignments or [])
    seeds = _resolve_seeds(name, query, mentions or {}, used_alignments)
    needles = list(dict.fromkeys([*seeds, (query or "").strip()]))
    items = _prefer_aligned_items(
        retrieve_chunks(" ".join(item for item in needles if item)[:200]),
        _aligned_names(used_alignments),
    )
    return _pack(name, seeds, items, _summarize(name, seeds, items), "ragflow", alignments=used_alignments)


def search_chroma_knowledge(
    *,
    query: str,
    handler: str = "",
    alignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """只查 Chroma 疾病/症状向量。有对齐名时丢掉对不上的片段。"""
    from algorithm.kg_query import _aligned_names, _chroma_snips, _pack, _prefer_aligned_items, _summarize

    name = (handler or "").strip() or "fallback"
    text = (query or "").strip()
    names = _aligned_names(alignments or [])
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import run_rag_pipeline

        if is_rag_enhanced():
            pack = run_rag_pipeline("chroma", query=text, handler=name)
            pack["items"] = _prefer_aligned_items(list(pack.get("items") or []), names)
            pack["summary"] = _summarize(name, names or ([text] if text else []), pack["items"])
            return pack
    except Exception:
        logger.warning("rag chroma pipeline failed", exc_info=True)
    items = _prefer_aligned_items(_chroma_snips(text), names)
    seeds = names or ([text] if len(text) >= 2 else [])
    return _pack(name, seeds, items, _summarize(name, seeds, items), "chroma", alignments=alignments or [])


def lookup_report_handbook(query: str) -> dict[str, Any]:
    q = (query or "").strip() or "化验单"
    packed = search_exam_handbook(query=q[:200], handler="report_upload")
    items = list(packed.get("items") or [])
    if not items:
        packed = search_chroma_knowledge(query=q[:200], handler="report_upload")
        items = list(packed.get("items") or [])
    return {"query": q, "kb_source": packed.get("source") or "", "items": items[:16]}
