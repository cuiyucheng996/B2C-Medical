"""记忆后端：接口对齐 Graphiti ingest/search。

Episode 仍落 Finding；事实另写 CompanionFact（不打 Episodic，不改 Consult*）。
"""

from __future__ import annotations

import logging
from typing import Any

from session import CompanionSession

logger = logging.getLogger(__name__)

__all__ = ["add_episode", "persist_session_findings", "search_facts"]


def add_episode(
    *,
    user_id: str,
    content: str,
    role_type: str = "user",
    source_description: str = "b2c_chat",
    said_at: str | None = None,
    extra_facts: list[dict[str, Any]] | None = None,
    extra_links: list[dict[str, Any]] | None = None,
    skip_entity: bool = False,
) -> dict[str, Any]:
    """写入长期 Finding。user_id = Neo4j UserInfo.id。"""
    uid = (user_id or "").strip()
    text = (content or "").strip()
    if not uid or not text:
        return {"ok": False, "reason": "empty"}
    if role_type == "assistant":
        return {"ok": True, "skipped": "assistant"}

    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.algorithm.user_memory import ingest_finding
    from algorithm.fact_extract import extract_companion_facts

    body = text
    if body.lower().startswith("user(") and ":" in body:
        body = body.split(":", 1)[1].strip() or text

    facts = list(extra_facts or [])
    facts.extend(extract_companion_facts(body, at=said_at, skip_entity=skip_entity))
    return ingest_finding(
        user_id=uid,
        raw_text=body,
        source="chat",
        kind="text",
        extra_facts=facts,
        extra_links=extra_links,
    )


def persist_session_findings(session: CompanionSession) -> dict[str, Any]:
    """会话结束时才把用户话抽离写入 Finding。过程中只用 session 短期消息。"""
    pending = [
        msg
        for msg in session.user
        if msg.get("role_type") == "user" and not msg.get("ingested")
    ]
    if not pending and session.persisted:
        return {"ok": True, "deferred": False, "skipped": "already_persisted", "ingested": 0, "items": []}
    from algorithm.fact_extract import facts_from_slot_memory, links_from_alignments, links_from_slot_memory

    items: list[dict[str, Any]] = []
    ingested = 0
    for msg in session.user:
        if msg.get("role_type") != "user" or msg.get("ingested"):
            continue
        extracted = bool(msg.get("entity_extracted"))
        mentions = msg.get("mentions") if isinstance(msg.get("mentions"), dict) else {}
        alignments = list(msg.get("alignments") or [])
        extra_facts = facts_from_slot_memory(None, mentions) if extracted else []
        extra_links = None
        if extracted:
            extra_links = links_from_slot_memory(None, mentions) + links_from_alignments(alignments)
        result = add_episode(
            user_id=session.group_id,
            content=str(msg.get("content") or ""),
            role_type="user",
            source_description="b2c_session_end",
            said_at=str(msg.get("timestamp") or "") or None,
            extra_facts=extra_facts,
            extra_links=extra_links,
            skip_entity=extracted,
        )
        items.append(result)
        if result.get("ok"):
            ingested += 1
            msg["ingested"] = True
    session.persisted = True
    return {
        "ok": True,
        "deferred": False,
        "ingested": ingested,
        "items": items,
        "user_id": session.group_id,
        "session_id": session.session_id,
    }


def search_facts(*, user_id: str, query: str, max_facts: int = 8) -> list[dict[str, Any]]:
    """按人检索仍有效的长期事实。"""
    uid = (user_id or "").strip()
    if not uid:
        return []

    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.algorithm.user_memory import load_user_memory_card

    card = load_user_memory_card(uid, limit=max(5, max_facts))
    facts: list[dict[str, Any]] = []
    q = (query or "").strip()
    for item in card.get("companion_facts") or []:
        name = str(item.get("name") or "").strip()
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        if q and q not in fact and q not in name and str(item.get("kind") or "") not in q:
            continue
        facts.append(
            {
                "name": item.get("kind") or name,
                "fact": ("已排除：" + name) if item.get("negated") else fact,
                "episode": item.get("id"),
            }
        )
        if len(facts) >= max_facts:
            return facts[:max_facts]
    summary = (card.get("summary") or "").strip()
    if summary:
        facts.append({"name": "current_profile", "fact": summary})
    for finding in card.get("findings") or []:
        raw = str(finding.get("raw_text") or "").strip()
        if not raw:
            continue
        if q and q not in raw and not any(
            str(item.get("name") or "") in q for item in (finding.get("extracts") or [])
        ):
            if len(facts) > 2:
                continue
        extracts = finding.get("extracts") or []
        if extracts:
            for item in extracts:
                facts.append(
                    {
                        "name": item.get("kind") or "extract",
                        "fact": f"{item.get('kind')}: {item.get('name') or item.get('mention')}",
                        "episode": finding.get("id"),
                    }
                )
        else:
            facts.append(
                {
                    "name": "utterance",
                    "fact": raw[:200],
                    "episode": finding.get("id"),
                }
            )
        if len(facts) >= max_facts:
            break
    return facts[:max_facts]
