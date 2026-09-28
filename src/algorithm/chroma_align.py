"""C 端：Chroma 粗召回 + DeepSeek 精排（不经 Milvus）。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["ChromaAlignRepository", "get_chroma_align_repository"]

EntityType = Literal["body_part", "disease", "symptom"]

ENTITY_COLLECTIONS: dict[str, list[str]] = {
    "body_part": ["consult_body_part"],
    "disease": ["disease"],
    "symptom": ["primary_symptom", "auxiliary_symptom"],
}


class ChromaAlignRepository:
    def get_standard_word_by_synonym(
        self,
        entity_type: EntityType,
        synonym: str,
        *,
        allowed_names: set[str] | None = None,
        max_distance: float = 0.5,
        utterance: str | None = None,
    ) -> tuple[str | None, float | None, str | None]:
        del max_distance
        text = (synonym or "").strip()
        if not text:
            return None, None, None

        ensure_uvgraph_root()
        from b2c_b2b_bridge import b2b_api_config
        from src.algorithm.chroma_search_core import search_entity_in_chroma_core

        collections = ENTITY_COLLECTIONS.get(entity_type) or []
        if not collections:
            return None, None, None

        recall_k = int(b2b_api_config().get_b2b_api_config().get("align_rerank_recall_k") or 10)
        try:
            hits = search_entity_in_chroma_core(
                text,
                collection_names=collections,
                allowed_names=set(allowed_names) if allowed_names else None,
                limit=max(recall_k, 1),
                max_distance=1.0,
            )
        except Exception as exc:
            logger.warning("Chroma 对齐失败: %s", exc)
            return None, None, None
        if not hits:
            logger.info("Chroma 未召回: type=%s mention=%s", entity_type, text)
            return None, None, None

        names = [str(row.get("standard_name") or "").strip() for row in hits]
        names = [name for name in names if name]
        from algorithm.align_rerank import rerank_align_candidates

        picked = rerank_align_candidates(text, names, entity_type=entity_type, utterance=utterance)
        if not picked:
            logger.warning("精排未选: type=%s mention=%s recall=%s", entity_type, text, names[:8])
            return None, None, None

        top = next((row for row in hits if str(row.get("standard_name") or "").strip() == picked), None)
        distance = top.get("distance") if top else None
        node_id = top.get("node_id") if top else None
        logger.info("Chroma+精排: %s %s -> %s distance=%s", entity_type, text, picked, distance)
        return picked, distance, str(node_id) if node_id is not None else None


@lru_cache(maxsize=1)
def get_chroma_align_repository() -> ChromaAlignRepository:
    return ChromaAlignRepository()
