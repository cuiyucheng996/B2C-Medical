"""多路召回：RRF 融合。"""

from __future__ import annotations

from algorithm.rag.types import RagHit

__all__ = ["rrf_merge"]


def rrf_merge(lists: list[list[RagHit]], *, k: int = 60) -> list[RagHit]:
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    scores: dict[str, float] = {}
    best: dict[str, RagHit] = {}
    for lst in lists:
        for rank, hit in enumerate(lst, start=1):
            key = (hit.text or "").strip()
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            prev = best.get(key)
            if prev is None or hit.score > prev.score:
                best[key] = hit
    merged: list[RagHit] = []
    for key, score in sorted(scores.items(), key=lambda item: -item[1]):
        row = best[key]
        merged.append(
            RagHit(
                text=row.text,
                score=round(score, 6),
                recall_rank=0,
                channel=row.channel,
                doc=row.doc,
                meta={**row.meta, "rrf_score": round(score, 6)},
            )
        )
    for idx, hit in enumerate(merged, start=1):
        hit.recall_rank = idx
    return merged
