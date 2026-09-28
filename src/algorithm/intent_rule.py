"""先大类、再细意图。规则关键词来自 data/intent_catalog.yaml。"""

from __future__ import annotations

from algorithm.intent_catalog import INTENT_BY_NAME
from algorithm.intent_manifest import load_intent_manifest

_manifest = load_intent_manifest()
INTENT_RULES: list[tuple[str, tuple[str, ...]]] = list(_manifest.rule_rules)
MAJOR_ORDER = _manifest.major_order

__all__ = ["INTENT_RULES", "MAJOR_ORDER", "match_all_intents", "match_intent", "match_major_intent"]


def _compact(question: str) -> str:
    return (question or "").replace(" ", "").lower()


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw.replace(" ", "").lower() in text for kw in keywords if kw)


def match_major_intent(question: str) -> str | None:
    """只判七个大类，不返回细名。"""
    text = _compact(question)
    if not text:
        return None
    buckets: dict[str, list[str]] = {name: [] for name in MAJOR_ORDER}
    for intent, keywords in INTENT_RULES:
        major = str((INTENT_BY_NAME.get(intent) or {}).get("major") or "")
        if major in buckets:
            buckets[major].extend(keywords)
    for major in MAJOR_ORDER:
        if _hit(text, tuple(buckets[major])):
            return major
    return None


def match_all_intents(question: str) -> list[str]:
    """跨大类收集所有关键词命中的细意图（用于多意图规则降级）。"""
    text = _compact(question)
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for intent, keywords in INTENT_RULES:
        if intent in seen:
            continue
        if _hit(text, keywords):
            seen.add(intent)
            found.append(intent)
    return found


def match_intent(question: str, *, major: str | None = None) -> str | None:
    """在已锁定的大类内选细意图；未给大类则不匹配。"""
    if not major or major == "other":
        return None
    text = _compact(question)
    if not text:
        return None
    for intent, keywords in INTENT_RULES:
        row = INTENT_BY_NAME.get(intent) or {}
        if str(row.get("major") or "") != major:
            continue
        if _hit(text, keywords):
            return intent
    return None
