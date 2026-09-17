"""检索后：关键词加权重排（与 kb_eval 一致）。"""

from __future__ import annotations

import re

from algorithm.rag.types import RagHit

__all__ = ["keyword_rerank_hits"]


def _query_terms(query: str) -> list[str]:
    compact = re.sub(r"\s+", "", query or "")
    terms: list[str] = []
    if compact:
        terms.append(compact)
        for size in (4, 3, 2):
            if len(compact) < size:
                continue
            terms.extend(compact[i : i + size] for i in range(0, len(compact) - size + 1))
    return list(dict.fromkeys(item for item in terms if len(item) >= 2))[:40]


def keyword_rerank_hits(query: str, hits: list[RagHit]) -> list[RagHit]:
    tokens = [tok for tok in _query_terms(query) if len(tok) >= 3]
    ranked: list[tuple[float, RagHit]] = []
    for hit in hits:
        text = hit.text
        kw = sum(text.count(tok) for tok in tokens)
        recall = float(hit.score or 0)
        score = round(recall + kw * 0.5, 4)
        meta = {**hit.meta, "keyword_hits": kw, "rerank_score": score}
        ranked.append(
            (
                score,
                RagHit(
                    text=hit.text,
                    score=score,
                    recall_rank=hit.recall_rank,
                    channel=hit.channel,
                    doc=hit.doc,
                    meta=meta,
                ),
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1].recall_rank))
    out: list[RagHit] = []
    for idx, (_, hit) in enumerate(ranked, start=1):
        moved = int(hit.recall_rank or 0) - idx if hit.recall_rank else 0
        hit.recall_rank = idx
        hit.meta["rerank_rank"] = idx
        hit.meta["moved"] = moved
        out.append(hit)
    return out
