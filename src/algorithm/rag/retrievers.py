"""分 channel 召回，输出 RagHit 列表。"""

from __future__ import annotations

from typing import Any

from algorithm.rag.types import RagHit
from algorithm.ragflow_kb import retrieve_hits

__all__ = ["retrieve_handbook_hits", "retrieve_chroma_hits", "hits_from_text_items"]


def hits_from_text_items(items: list[str], *, channel: str, source_score: float = 0.0) -> list[RagHit]:
    out: list[RagHit] = []
    for idx, text in enumerate(items, start=1):
        line = str(text or "").strip()
        if not line:
            continue
        out.append(
            RagHit(
                text=line,
                score=source_score,
                recall_rank=idx,
                channel=channel,
            )
        )
    return out


def retrieve_handbook_hits(query: str, *, top_k: int) -> list[RagHit]:
    rows = retrieve_hits(query, top_k=top_k)
    out: list[RagHit] = []
    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        out.append(
            RagHit(
                text=text,
                score=float(row.get("score") or 0),
                recall_rank=idx,
                channel="handbook",
                doc=str(row.get("doc") or row.get("document_name") or ""),
                meta={k: v for k, v in row.items() if k not in {"text", "score", "doc", "document_name"}},
            )
        )
    return out


def retrieve_chroma_hits(query: str) -> list[RagHit]:
    from algorithm.kg_query import _chroma_snips

    lines = _chroma_snips(query)
    return hits_from_text_items(lines, channel="chroma")
