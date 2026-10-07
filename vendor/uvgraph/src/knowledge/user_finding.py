"""UserInfo 长期记忆入 Neo4j：Finding 原始桶 + EXTRACTED 到问诊知识节点。

关系名与批同步一致用大写 HAVE。不对齐的 span 只写在 Finding 属性上，不新建病名节点。
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any
from uuid import uuid4

from src.configs import NEO4J_CONFIG
from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver
from src.schemas.user_memory import CompanionFactRecord, ExtractedLink, FindingRecord

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_finding_indexes",
    "find_companion_facts",
    "link_extracts",
    "list_companion_facts",
    "list_user_findings",
    "load_current_facts",
    "merge_companion_facts",
    "merge_finding",
    "set_current_stated",
]


def _db() -> str:
    return str(NEO4J_CONFIG.get("database") or "neo4j")


def _driver():
    return get_neo4j_driver()


@lru_cache
def ensure_finding_indexes() -> None:
    statements = (
        """
        CREATE CONSTRAINT finding_content_hash IF NOT EXISTS
        FOR (f:Finding) REQUIRE f.content_hash IS UNIQUE
        """,
        """
        CREATE INDEX finding_user_created IF NOT EXISTS
        FOR (f:Finding) ON (f.created_at)
        """,
        """
        CREATE CONSTRAINT companion_fact_hash IF NOT EXISTS
        FOR (c:CompanionFact) REQUIRE c.fact_hash IS UNIQUE
        """,
    )
    driver = _driver()
    db = _db()
    for cypher in statements:
        try:
            driver.execute_query(cypher, database_=db)
        except Exception:
            logger.warning("Finding 约束/索引创建失败，继续写入: %s", cypher.split()[1:4])


def _match_or_stub_user(user_id: str, *, created_at: str) -> None:
    uid = (user_id or "").strip()
    found = _driver().execute_query(
        """
        MATCH (u:UserInfo)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        RETURN u.id AS id LIMIT 1
        """,
        parameters_={"user_id": uid},
        database_=_db(),
    )
    if found.records:
        return
    stub_id: Any = int(uid) if uid.isdigit() else uid
    _driver().execute_query(
        """
        MERGE (u:UserInfo {id: $id})
        ON CREATE SET u.source = 'memory_stub', u.create_time = $created_at
        """,
        parameters_={"id": stub_id, "created_at": created_at},
        database_=_db(),
    )


def merge_finding(
    *,
    user_id: str,
    content_hash: str,
    finding_id: str,
    kind: str,
    raw_text: str,
    image_url: str,
    thread_id: str,
    source: str,
    created_at: str,
    mentions: list[str] | None = None,
) -> str:
    """MERGE UserInfo（按 toString(id)）+ Finding；已存在则只补属性。"""
    ensure_finding_indexes()
    uid = (user_id or "").strip()
    _match_or_stub_user(uid, created_at=created_at)
    mentions_text = " | ".join(mentions or [])
    _driver().execute_query(
        """
        MATCH (u:UserInfo)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        WITH u LIMIT 1
        MERGE (f:Finding {content_hash: $content_hash})
        ON CREATE SET
          f.id = $finding_id,
          f.kind = $kind,
          f.raw_text = $raw_text,
          f.image_url = $image_url,
          f.thread_id = $thread_id,
          f.source = $source,
          f.created_at = $created_at,
          f.mentions = $mentions
        ON MATCH SET
          f.raw_text = coalesce(f.raw_text, $raw_text),
          f.image_url = coalesce(f.image_url, $image_url)
        MERGE (u)-[:HAVE]->(f)
        RETURN f.id AS id
        """,
        parameters_={
            "user_id": uid,
            "content_hash": content_hash,
            "finding_id": finding_id,
            "kind": kind,
            "raw_text": raw_text or "",
            "image_url": image_url or "",
            "thread_id": thread_id or "",
            "source": source,
            "created_at": created_at,
            "mentions": mentions_text,
        },
        database_=_db(),
    )
    return finding_id


def link_extracts(*, content_hash: str, links: list[ExtractedLink]) -> int:
    """抽离结果连到已有 Consult* 节点；找不到标准节点则跳过。"""
    if not links:
        return 0
    linked = 0
    driver = _driver()
    db = _db()
    for link in links:
        kind = str(link.get("kind") or "")
        name = str(link.get("name") or "").strip()
        mention = str(link.get("mention") or name)
        if not name:
            continue
        if kind == "body_part":
            cypher = """
            MATCH (f:Finding {content_hash: $content_hash})
            MATCH (n:ConsultBodyPart)
            WHERE n.part_name = $name OR toString(n.id) = $node_id
            WITH f, n LIMIT 1
            MERGE (f)-[e:EXTRACTED {kind: 'body_part'}]->(n)
            SET e.mention = $mention, e.name = $name, e.aligned = true
            RETURN count(e) AS c
            """
        elif kind == "disease":
            cypher = """
            MATCH (f:Finding {content_hash: $content_hash})
            MATCH (n:ConsultDisease)
            WHERE n.disease_name = $name OR toString(n.id) = $node_id
            WITH f, n LIMIT 1
            MERGE (f)-[e:EXTRACTED {kind: 'disease'}]->(n)
            SET e.mention = $mention, e.name = $name, e.aligned = true
            RETURN count(e) AS c
            """
        elif kind == "symptom":
            cypher = """
            MATCH (f:Finding {content_hash: $content_hash})
            MATCH (n:ConsultPrimarySymptom)
            WHERE n.symptom_name = $name OR toString(n.id) = $node_id
            WITH f, n LIMIT 1
            MERGE (f)-[e:EXTRACTED {kind: 'symptom'}]->(n)
            SET e.mention = $mention, e.name = $name, e.aligned = true
            RETURN count(e) AS c
            """
        else:
            continue
        result = driver.execute_query(
            cypher,
            parameters_={
                "content_hash": content_hash,
                "name": name,
                "node_id": str(link.get("node_id") or ""),
                "mention": mention,
            },
            database_=db,
        )
        count = int(result.records[0]["c"]) if result.records else 0
        if count:
            linked += 1
        else:
            logger.info("未对齐知识节点 kind=%s name=%s", kind, name)
    return linked


def merge_companion_facts(
    *,
    user_id: str,
    content_hash: str,
    facts: list[dict[str, Any]],
    said_at: str,
) -> int:
    """Graphiti 式事实：CompanionFact + STATED_FACT / HAS_FACT。不碰 Consult*。"""
    uid = (user_id or "").strip()
    if not uid or not facts:
        return 0
    ensure_finding_indexes()
    written = 0
    driver = _driver()
    db = _db()
    for item in facts:
        name = str(item.get("name") or "").strip()
        fact = str(item.get("fact") or name).strip()
        kind = str(item.get("kind") or "other").strip() or "other"
        if not name or not fact:
            continue
        negated = bool(item.get("negated"))
        fact_hash = hashlib.sha256(f"{uid}|{kind}|{name}".encode("utf-8")).hexdigest()
        fact_id = str(uuid4())
        try:
            driver.execute_query(
                """
                MATCH (u:UserInfo)
                WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
                WITH u LIMIT 1
                MERGE (c:CompanionFact {fact_hash: $fact_hash})
                ON CREATE SET
                  c.id = $fact_id,
                  c.name = $name,
                  c.kind = $kind,
                  c.user_id = $user_id
                SET c.fact = $fact,
                    c.negated = $negated,
                    c.last_said = $said_at
                WITH u, c
                MERGE (u)-[r:STATED_FACT]->(c)
                SET r.status = 'current', r.last_said = $said_at, r.negated = $negated
                WITH c
                MATCH (f:Finding {content_hash: $content_hash})
                MERGE (f)-[:HAS_FACT]->(c)
                """,
                parameters_={
                    "user_id": uid,
                    "fact_hash": fact_hash,
                    "fact_id": fact_id,
                    "name": name,
                    "kind": kind,
                    "fact": fact,
                    "negated": negated,
                    "said_at": said_at,
                    "content_hash": content_hash,
                },
                database_=db,
            )
            written += 1
        except Exception:
            logger.exception("merge companion fact failed name=%s", name)
    return written


def list_companion_facts(user_id: str, *, limit: int = 20) -> list[CompanionFactRecord]:
    uid = (user_id or "").strip()
    if not uid:
        return []
    result = _driver().execute_query(
        """
        MATCH (u:UserInfo)-[r:STATED_FACT {status: 'current'}]->(c:CompanionFact)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        RETURN c.id AS id,
               c.fact_hash AS fact_hash,
               c.name AS name,
               c.fact AS fact,
               c.kind AS kind,
               coalesce(c.negated, false) AS negated,
               c.last_said AS last_said
        ORDER BY c.last_said DESC
        LIMIT $limit
        """,
        parameters_={"user_id": uid, "limit": int(limit)},
        database_=_db(),
    )
    rows: list[CompanionFactRecord] = []
    for rec in result.records:
        rows.append(
            {
                "id": rec.get("id") or "",
                "fact_hash": rec.get("fact_hash") or "",
                "name": rec.get("name") or "",
                "fact": rec.get("fact") or "",
                "kind": rec.get("kind") or "other",
                "negated": bool(rec.get("negated")),
                "last_said": rec.get("last_said") or "",
            }
        )
    return rows


def find_companion_facts(
    user_id: str,
    names: list[str],
    *,
    limit: int = 12,
) -> list[CompanionFactRecord]:
    """按名字筛当前有效事实，给实体记忆中间件一次查清。"""
    uid = (user_id or "").strip()
    wanted = [str(name).strip() for name in names or [] if str(name).strip()]
    if not uid or not wanted:
        return []
    result = _driver().execute_query(
        """
        MATCH (u:UserInfo)-[r:STATED_FACT {status: 'current'}]->(c:CompanionFact)
        WHERE (toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id)
          AND c.name IN $names
        RETURN c.id AS id,
               c.fact_hash AS fact_hash,
               c.name AS name,
               c.fact AS fact,
               c.kind AS kind,
               coalesce(c.negated, false) AS negated,
               c.last_said AS last_said
        ORDER BY c.last_said DESC
        LIMIT $limit
        """,
        parameters_={"user_id": uid, "names": wanted, "limit": int(limit)},
        database_=_db(),
    )
    rows: list[CompanionFactRecord] = []
    for rec in result.records:
        rows.append(
            {
                "id": rec.get("id") or "",
                "fact_hash": rec.get("fact_hash") or "",
                "name": rec.get("name") or "",
                "fact": rec.get("fact") or "",
                "kind": rec.get("kind") or "other",
                "negated": bool(rec.get("negated")),
                "last_said": rec.get("last_said") or "",
            }
        )
    return rows


def set_current_stated(
    *,
    user_id: str,
    kind: str,
    name: str,
    finding_hash: str,
    said_at: str,
) -> None:
    """UserInfo 当前画像边：旧 current 标 superseded，再 MERGE 新边。"""
    uid = (user_id or "").strip()
    name = (name or "").strip()
    if not uid or not name:
        return
    if kind == "body_part":
        rel, label, name_prop = "STATED_PART", "ConsultBodyPart", "part_name"
    elif kind == "disease":
        rel, label, name_prop = "STATED_DISEASE", "ConsultDisease", "disease_name"
    elif kind == "symptom":
        rel, label, name_prop = "STATED_SYMPTOM", "ConsultPrimarySymptom", "symptom_name"
    else:
        return
    _driver().execute_query(
        f"""
        MATCH (u:UserInfo)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        OPTIONAL MATCH (u)-[old:`{rel}`]->()
        WHERE old.status = 'current'
        SET old.status = 'superseded'
        WITH u
        MATCH (n:`{label}`)
        WHERE n.{name_prop} = $name
        WITH u, n LIMIT 1
        MERGE (u)-[r:`{rel}`]->(n)
        SET r.status = 'current',
            r.last_said = $said_at,
            r.finding_hash = $finding_hash,
            r.name = $name
        """,
        parameters_={
            "user_id": uid,
            "name": name,
            "said_at": said_at,
            "finding_hash": finding_hash,
        },
        database_=_db(),
    )


def load_current_facts(user_id: str) -> dict[str, Any]:
    uid = (user_id or "").strip()
    result = _driver().execute_query(
        """
        MATCH (u:UserInfo)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        OPTIONAL MATCH (u)-[rp:STATED_PART {status: 'current'}]->(p:ConsultBodyPart)
        OPTIONAL MATCH (u)-[rd:STATED_DISEASE {status: 'current'}]->(d:ConsultDisease)
        OPTIONAL MATCH (u)-[rs:STATED_SYMPTOM {status: 'current'}]->(s:ConsultPrimarySymptom)
        RETURN
          collect(DISTINCT p.part_name) AS parts,
          collect(DISTINCT d.disease_name) AS diseases,
          collect(DISTINCT s.symptom_name) AS symptoms
        """,
        parameters_={"user_id": uid},
        database_=_db(),
    )
    if not result.records:
        return {"body_part": None, "disease": None, "symptoms": []}
    row = result.records[0]
    parts = [x for x in (row.get("parts") or []) if x]
    diseases = [x for x in (row.get("diseases") or []) if x]
    symptoms = [x for x in (row.get("symptoms") or []) if x]
    return {
        "body_part": parts[0] if parts else None,
        "disease": diseases[0] if diseases else None,
        "symptoms": symptoms,
    }


def list_user_findings(user_id: str, *, limit: int = 30) -> list[FindingRecord]:
    uid = (user_id or "").strip()
    result = _driver().execute_query(
        """
        MATCH (u:UserInfo)-[:HAVE]->(f:Finding)
        WHERE toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id
        OPTIONAL MATCH (f)-[e:EXTRACTED]->(n)
        WITH f, collect({
          kind: e.kind,
          name: e.name,
          mention: e.mention,
          aligned: e.aligned,
          label: labels(n)[0]
        }) AS extracts
        RETURN f.id AS id,
               f.content_hash AS content_hash,
               f.kind AS kind,
               f.raw_text AS raw_text,
               f.image_url AS image_url,
               f.thread_id AS thread_id,
               f.source AS source,
               f.created_at AS created_at,
               extracts
        ORDER BY f.created_at DESC
        LIMIT $limit
        """,
        parameters_={"user_id": uid, "limit": int(limit)},
        database_=_db(),
    )
    items: list[FindingRecord] = []
    for rec in result.records:
        extracts = []
        for item in rec.get("extracts") or []:
            if not item or not item.get("kind"):
                continue
            extracts.append(item)
        items.append(
            {
                "id": rec.get("id") or "",
                "content_hash": rec.get("content_hash") or "",
                "kind": rec.get("kind") or "text",
                "raw_text": rec.get("raw_text") or "",
                "image_url": rec.get("image_url") or "",
                "thread_id": rec.get("thread_id") or "",
                "source": rec.get("source") or "chat",
                "created_at": rec.get("created_at") or "",
                "extracts": extracts,
            }
        )
    return items
