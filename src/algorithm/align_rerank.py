"""Chroma 粗召回后的精排：B2B 实现经桥接暴露。"""

from __future__ import annotations

from typing import Any, Literal

from b2c_b2b_bridge import b2b_align_rerank

EntityType = Literal["body_part", "disease", "symptom"]

__all__ = ["EntityType", "rerank_align_candidates"]


def rerank_align_candidates(
    mention: str,
    candidates: list[str],
    *,
    entity_type: EntityType,
    utterance: str | None = None,
) -> str | None:
    fn: Any = b2b_align_rerank().rerank_align_candidates
    return fn(mention, candidates, entity_type=entity_type, utterance=utterance)
