"""C 端实体对齐：同义词表 + Chroma 粗召回 + DeepSeek 精排。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from algorithm.chroma_align import get_chroma_align_repository

logger = logging.getLogger(__name__)

__all__ = ["AlignHit", "entity_align", "EMBED_ACCEPT_DISTANCE"]

EntityType = Literal["body_part", "disease", "symptom"]
EMBED_ACCEPT_DISTANCE = 0.5


@dataclass(frozen=True)
class AlignHit:
    mention: str
    standard_name: str | None
    node_id: int | None
    node_label: str | None
    method: str
    score: float
    needs_review: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.standard_name) and not self.needs_review


def _lookup_synonym(entity_type: EntityType, text: str) -> str | None:
    from b2c_b2b_bridge import b2b_entity_synonyms

    return b2b_entity_synonyms().lookup_synonym(entity_type, text)


def _name_of(row: dict[str, Any], name_key: str) -> str:
    return str(row.get(name_key) or "").strip()


def _row_id(row: dict[str, Any], name_key: str) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    return _name_of(row, name_key)


def _find_row(catalog: list[dict[str, Any]], name_key: str, name: str) -> dict[str, Any] | None:
    target = (name or "").strip()
    if not target:
        return None
    for row in catalog:
        if _name_of(row, name_key) == target:
            return row
    for row in catalog:
        std = _name_of(row, name_key)
        if std and (target in std or std in target):
            return row
    return None


def _find_row_by_id(catalog: list[dict[str, Any]], name_key: str, node_id: str | None) -> dict[str, Any] | None:
    if not node_id:
        return None
    for row in catalog:
        if _row_id(row, name_key) == str(node_id):
            return row
    return None


def _parse_node_id(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _allowed_names(catalog: list[dict[str, Any]], name_key: str) -> set[str]:
    return {_name_of(row, name_key) for row in catalog if _name_of(row, name_key)}


def entity_align(
    entity_type: EntityType,
    mention: str,
    catalog: list[dict[str, Any]],
    *,
    name_key: str,
    node_label: str,
    for_span: bool = False,
    utterance: str | None = None,
) -> AlignHit:
    text = (mention or "").strip()
    if not text:
        return AlignHit(text, None, None, node_label, "empty", 0.0, True)

    alias = _lookup_synonym(entity_type, text)
    if alias:
        row = _find_row(catalog, name_key, alias)
        if row:
            logger.info("同义词表: %s -> %s", text, _name_of(row, name_key))
            return AlignHit(
                mention=text,
                standard_name=_name_of(row, name_key),
                node_id=row.get("id") if isinstance(row.get("id"), int) else _parse_node_id(str(row.get("id"))),
                node_label=node_label,
                method="synonym",
                score=1.0,
            )

    exact = _find_row(catalog, name_key, text)
    if exact:
        return AlignHit(
            text,
            _name_of(exact, name_key),
            exact.get("id") if isinstance(exact.get("id"), int) else _parse_node_id(_row_id(exact, name_key)),
            node_label,
            "exact",
            1.0,
        )

    too_long = len(text) > 24 if for_span else (len(text) > 12 or any(mark in text for mark in "。！？，、呢吧啊呀"))
    if too_long:
        return AlignHit(text, None, None, node_label, "skip_chroma", 0.0, True)

    allowed = _allowed_names(catalog, name_key) or None
    try:
        std_name, distance, raw_id = get_chroma_align_repository().get_standard_word_by_synonym(
            entity_type,
            text,
            allowed_names=allowed,
            max_distance=EMBED_ACCEPT_DISTANCE,
            utterance=utterance,
        )
    except Exception as exc:
        logger.warning("Chroma 对齐失败: %s", exc)
        return AlignHit(text, None, None, node_label, "miss", 0.0, True)

    row = _find_row_by_id(catalog, name_key, raw_id) or _find_row(catalog, name_key, std_name or "")
    node_id = row.get("id") if row and isinstance(row.get("id"), int) else _parse_node_id(raw_id)
    standard = _name_of(row, name_key) if row else std_name
    score = 0.9 if standard else 0.0

    if standard:
        return AlignHit(
            mention=text,
            standard_name=standard,
            node_id=node_id,
            node_label=node_label,
            method="chroma_rerank",
            score=score,
        )

    return AlignHit(
        mention=text,
        standard_name=standard,
        node_id=node_id,
        node_label=node_label,
        method="chroma",
        score=score,
        needs_review=True,
    )
