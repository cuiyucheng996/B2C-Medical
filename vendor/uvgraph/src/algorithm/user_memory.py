"""B2C 长期记忆：从用户原文/图片建 Finding，抽离后写入 Neo4j。"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from src.configs.medical_api import get_medical_api_config
from src.knowledge.body_disease_map import BODY_PARTS, DISEASES_BY_BODY
from src.schemas.user_memory import ExtractedLink, FindingKind, FindingSource, UserMemoryCard

logger = logging.getLogger(__name__)

__all__ = [
    "ingest_consult_state_memory",
    "ingest_finding",
    "links_from_text",
    "load_user_memory_card",
    "load_user_memory_summary",
    "user_memory_enabled",
]


def user_memory_enabled() -> bool:
    return bool(get_medical_api_config().get("user_memory_enabled", True))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _content_hash(
    user_id: str,
    *,
    kind: str,
    raw_text: str,
    image_url: str,
) -> str:
    raw = "|".join(
        [
            (user_id or "").strip(),
            kind,
            (raw_text or "").strip(),
            (image_url or "").strip(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _all_disease_names() -> list[str]:
    names: list[str] = []
    for items in DISEASES_BY_BODY.values():
        names.extend(items)
    return list(dict.fromkeys(names))


def links_from_text(text: str) -> list[ExtractedLink]:
    """规则抽离：原文里出现的标准部位/病名。不新建知识节点。"""
    compact = (text or "").replace(" ", "")
    if not compact:
        return []
    links: list[ExtractedLink] = []
    for part in BODY_PARTS:
        if part != "其他" and part in compact:
            links.append({"kind": "body_part", "name": part, "mention": part, "label": "ConsultBodyPart"})
    for name in _all_disease_names():
        if name and name in compact:
            links.append({"kind": "disease", "name": name, "mention": name, "label": "ConsultDisease"})
    return links


def _human_texts(state: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for msg in state.get("messages") or []:
        msg_type = getattr(msg, "type", "") or ""
        if msg_type != "human":
            continue
        content = msg.content
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if text:
            texts.append(text)
    return texts


def _confirmed_links(state: Mapping[str, Any]) -> list[ExtractedLink]:
    from src.algorithm.round1 import body_from_state, disease_from_state
    from src.algorithm.round2 import symptoms_from_state

    links: list[ExtractedLink] = []
    body = body_from_state(state)
    if body:
        links.append({"kind": "body_part", "name": body, "mention": body, "label": "ConsultBodyPart"})
    disease = disease_from_state(state)
    if disease:
        links.append({"kind": "disease", "name": disease, "mention": disease, "label": "ConsultDisease"})
    for symptom in symptoms_from_state(state) or []:
        name = str(symptom).strip()
        if name:
            links.append({"kind": "symptom", "name": name, "mention": name, "label": "ConsultPrimarySymptom"})
    confirmed = state.get("confirmed_symptom") or state.get("selected_symptom")
    if confirmed:
        name = str(confirmed).strip()
        if name and not any(item.get("name") == name for item in links if item.get("kind") == "symptom"):
            links.append({"kind": "symptom", "name": name, "mention": name, "label": "ConsultPrimarySymptom"})
    return links


def ingest_finding(
    *,
    user_id: str,
    raw_text: str = "",
    image_url: str = "",
    thread_id: str = "",
    kind: FindingKind | None = None,
    source: FindingSource = "chat",
    extra_links: list[ExtractedLink] | None = None,
    extra_facts: list[dict[str, Any]] | None = None,
    update_stated: bool = False,
) -> dict[str, Any]:
    uid = (user_id or "").strip()
    text = (raw_text or "").strip()
    image = (image_url or "").strip()
    if not uid:
        return {"ok": False, "reason": "missing_user_id"}
    if not text and not image:
        return {"ok": False, "reason": "empty_finding"}
    if not user_memory_enabled():
        return {"ok": False, "reason": "disabled"}

    resolved_kind: FindingKind = kind or ("image" if image and not text else "text")
    content_hash = _content_hash(
        uid, kind=resolved_kind, raw_text=text, image_url=image
    )
    created_at = _now()
    finding_id = str(uuid4())
    if extra_links is None:
        links = links_from_text(text)
    else:
        links = list(extra_links)
    mentions = [str(item.get("mention") or item.get("name") or "") for item in links]

    try:
        from src.knowledge.user_finding import link_extracts, merge_companion_facts, merge_finding, set_current_stated

        merge_finding(
            user_id=uid,
            content_hash=content_hash,
            finding_id=finding_id,
            kind=resolved_kind,
            raw_text=text,
            image_url=image,
            thread_id=thread_id,
            source=source,
            created_at=created_at,
            mentions=mentions,
        )
        linked = link_extracts(content_hash=content_hash, links=links)
        fact_count = 0
        facts = list(extra_facts or [])
        for item in links:
            name = str(item.get("name") or item.get("mention") or "").strip()
            kind = str(item.get("kind") or "other")
            if not name:
                continue
            facts.append(
                {
                    "name": name,
                    "fact": f"用户提到{kind}：{name}",
                    "kind": "other",
                    "negated": False,
                }
            )
        if facts:
            fact_count = merge_companion_facts(
                user_id=uid,
                content_hash=content_hash,
                facts=facts,
                said_at=created_at,
            )
        if update_stated:
            for item in extra_links or []:
                set_current_stated(
                    user_id=uid,
                    kind=str(item.get("kind") or ""),
                    name=str(item.get("name") or ""),
                    finding_hash=content_hash,
                    said_at=created_at,
                )
        return {
            "ok": True,
            "finding_id": finding_id,
            "content_hash": content_hash,
            "linked": linked,
            "facts": fact_count,
        }
    except Exception:
        logger.exception("ingest_finding failed user_id=%s", uid)
        return {"ok": False, "reason": "neo4j_error"}


def ingest_consult_state_memory(state: Mapping[str, Any], *, thread_id: str = "") -> dict[str, Any]:
    """会话落库时：每条用户原话建 Finding；锁定的部位/病/症更新当前画像。"""
    uid = str(state.get("user_id") or "").strip()
    if not uid or not user_memory_enabled():
        return {"ok": False, "ingested": 0}

    tid = thread_id or str(state.get("thread_id") or "")
    ingested = 0
    last_hash = ""
    confirmed = _confirmed_links(state)
    for text in _human_texts(state):
        result = ingest_finding(
            user_id=uid,
            raw_text=text,
            thread_id=tid,
            kind="text",
            source="chat",
        )
        if result.get("ok"):
            ingested += 1
            last_hash = str(result.get("content_hash") or last_hash)

    image_url = str(state.get("finding_image_url") or "").strip()
    if image_url:
        result = ingest_finding(
            user_id=uid,
            image_url=image_url,
            thread_id=tid,
            kind="image",
            source="upload",
        )
        if result.get("ok"):
            ingested += 1
            last_hash = str(result.get("content_hash") or last_hash)

    if confirmed:
        snapshot_text = "；".join(
            f"{item['kind']}={item['name']}" for item in confirmed if item.get("name")
        )
        result = ingest_finding(
            user_id=uid,
            raw_text=snapshot_text,
            thread_id=tid,
            kind="consult_state",
            source="consult",
            extra_links=confirmed,
            update_stated=True,
        )
        if result.get("ok"):
            ingested += 1
            last_hash = str(result.get("content_hash") or last_hash)

    return {"ok": True, "ingested": ingested, "content_hash": last_hash}


def load_user_memory_card(user_id: str, *, limit: int = 20) -> UserMemoryCard:
    uid = (user_id or "").strip()
    facts = {"body_part": None, "disease": None, "symptoms": []}
    findings = []
    companion_facts: list = []
    try:
        from src.knowledge.user_finding import list_companion_facts, list_user_findings, load_current_facts

        facts = load_current_facts(uid)
        findings = list_user_findings(uid, limit=limit)
        companion_facts = list_companion_facts(uid, limit=limit)
    except Exception:
        logger.exception("load_user_memory_card failed user_id=%s", uid)
    summary = _format_summary(facts, companion_facts)
    return {
        "user_id": uid,
        "summary": summary,
        "current_body_part": facts.get("body_part"),
        "current_disease": facts.get("disease"),
        "current_symptoms": list(facts.get("symptoms") or []),
        "findings": findings,
        "companion_facts": companion_facts,
    }


def load_user_memory_summary(user_id: str) -> str:
    return load_user_memory_card(user_id).get("summary") or ""


def _format_summary(facts: Mapping[str, Any], companion_facts: list | None = None) -> str:
    parts: list[str] = []
    if facts.get("body_part"):
        parts.append(f"部位 {facts['body_part']}")
    if facts.get("disease"):
        parts.append(f"疾病 {facts['disease']}")
    symptoms = [str(item) for item in (facts.get("symptoms") or []) if item]
    if symptoms:
        parts.append("症状 " + "、".join(symptoms))
    for item in companion_facts or []:
        text = str(item.get("fact") or item.get("name") or "").strip()
        if not text:
            continue
        if item.get("negated"):
            parts.append("已排除 " + str(item.get("name") or text))
        else:
            parts.append(text)
        if len(parts) >= 12:
            break
    if not parts:
        return ""
    return "用户长期记忆（已确认）：" + "；".join(parts)
