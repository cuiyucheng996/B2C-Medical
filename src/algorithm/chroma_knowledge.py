"""C 端疾病/症状向量检索（仅 Chroma）。"""

from __future__ import annotations

import logging
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["chroma_knowledge_snippets"]


def chroma_knowledge_snippets(query: str, *, limit: int = 4) -> list[str]:
    text = (query or "").strip()
    if not text:
        return []
    try:
        ensure_uvgraph_root()
        from src.algorithm.chroma_search_core import search_diseases_in_chroma_core

        hits = search_diseases_in_chroma_core(text, limit=limit) or []
    except Exception as exc:
        logger.warning("Chroma 知识检索失败: %s", exc)
        return []
    lines: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        name = str(hit.get("name") or hit.get("disease") or hit.get("title") or "").strip()
        snippet = str(hit.get("text") or hit.get("content") or hit.get("metadata") or "").strip()
        if name and snippet:
            lines.append(f"{name}：{snippet[:80]}")
        elif name:
            lines.append(name)
        elif snippet:
            lines.append(snippet[:80])
    return lines[:6]
