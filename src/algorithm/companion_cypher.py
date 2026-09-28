"""C 端 Cypher：用户 Finding + 对齐到的知识节点。不做证型缩窄。"""

from __future__ import annotations

import logging
import time
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "align_catalog_cache_status",
    "invalidate_align_catalog_cache",
    "list_align_catalogs",
    "query_companion_graph",
    "warm_align_catalogs",
]

# 与 Neo4j 目录一致；TTL 内复用快照，对齐逻辑不变
_ALIGN_CATALOG_TTL_SEC = 600
_catalog_cache: dict[str, list[dict[str, Any]]] | None = None
_catalog_cached_at: float = 0.0


def invalidate_align_catalog_cache() -> None:
    global _catalog_cache, _catalog_cached_at
    _catalog_cache = None
    _catalog_cached_at = 0.0


def warm_align_catalogs() -> int:
    """启动时拉一次目录进缓存。返回耗时 ms。"""
    t0 = time.perf_counter()
    list_align_catalogs()
    return int((time.perf_counter() - t0) * 1000)


def align_catalog_cache_status() -> dict[str, Any]:
    """评测台：目录缓存是否命中、TTL 与条目数。"""
    now = time.monotonic()
    if _catalog_cache is None:
        return {
            "ready": False,
            "cached": False,
            "ttl_sec": _ALIGN_CATALOG_TTL_SEC,
            "age_sec": None,
            "counts": {"body_part": 0, "disease": 0, "symptom": 0},
        }
    age = max(0.0, now - _catalog_cached_at)
    counts = {
        "body_part": len(_catalog_cache.get("body_part") or []),
        "disease": len(_catalog_cache.get("disease") or []),
        "symptom": len(_catalog_cache.get("symptom") or []),
    }
    return {
        "ready": age < _ALIGN_CATALOG_TTL_SEC and any(counts.values()),
        "cached": True,
        "ttl_sec": _ALIGN_CATALOG_TTL_SEC,
        "age_sec": int(age),
        "counts": counts,
    }


def list_align_catalogs() -> dict[str, list[dict[str, Any]]]:
    """给对齐用的入口节点表：部位 / 疾病 / 主症。失败则空表，对齐仍可走 span。"""
    global _catalog_cache, _catalog_cached_at
    now = time.monotonic()
    if _catalog_cache is not None and (now - _catalog_cached_at) < _ALIGN_CATALOG_TTL_SEC:
        return _catalog_cache
    loaded = _load_align_catalogs_from_neo4j()
    _catalog_cache = loaded
    _catalog_cached_at = now
    return loaded


def _load_align_catalogs_from_neo4j() -> dict[str, list[dict[str, Any]]]:
    ensure_uvgraph_root()
    empty: dict[str, list[dict[str, Any]]] = {"body_part": [], "disease": [], "symptom": []}
    try:
        from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver
        from src.configs import NEO4J_CONFIG

        driver = get_neo4j_driver()
        db = str(NEO4J_CONFIG.get("database") or "neo4j")
    except Exception:
        logger.exception("companion catalog driver failed")
        return empty

    return {
        "body_part": _query_rows(
            driver,
            db,
            """
            MATCH (p:ConsultBodyPart)
            WHERE coalesce(p.status, 1) = 1
            RETURN p.id AS id, p.part_name AS part_name
            ORDER BY coalesce(p.sort_order, 0), p.part_name
            """,
        ),
        "disease": _query_rows(
            driver,
            db,
            """
            MATCH (d:ConsultDisease)
            WHERE coalesce(d.status, 1) = 1
            RETURN d.id AS id, d.disease_name AS disease_name
            ORDER BY d.disease_name
            """,
        ),
        "symptom": _query_rows(
            driver,
            db,
            """
            MATCH (s:ConsultPrimarySymptom)
            WHERE coalesce(s.status, 1) = 1
            RETURN s.id AS id, s.symptom_name AS symptom_name
            ORDER BY s.symptom_name
            """,
        ),
    }


def query_companion_graph(
    *,
    user_id: str,
    alignments: list[dict[str, Any]],
    extra_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """读长期记忆，确认知识库节点，并按对齐名检索用户 Finding。不刷证型。"""
    ensure_uvgraph_root()
    uid = (user_id or "").strip()
    memory: dict[str, Any] = {}
    try:
        from src.algorithm.user_memory import load_user_memory_card

        card = load_user_memory_card(uid, limit=8)
        memory = {
            "summary": card.get("summary") or "",
            "current_body_part": card.get("current_body_part"),
            "current_disease": card.get("current_disease"),
            "current_symptoms": list(card.get("current_symptoms") or []),
        }
    except Exception:
        logger.exception("companion cypher memory failed user_id=%s", uid)

    knowledge: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    try:
        from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver
        from src.configs import NEO4J_CONFIG

        driver = get_neo4j_driver()
        db = str(NEO4J_CONFIG.get("database") or "neo4j")
        seen: set[str] = set()
        names: list[str] = []
        for item in alignments:
            node_id = item.get("node_id")
            label = str(item.get("node_label") or "")
            name = str(item.get("standard_name") or item.get("mention") or "").strip()
            key = f"{label}:{node_id or name}"
            if name:
                names.append(name)
            if key in seen or (not node_id and not name):
                continue
            seen.add(key)
            row = _lookup_node(driver, db, label=label, node_id=node_id, name=name)
            if row:
                knowledge.append(row)
        for extra in extra_keywords or []:
            text = str(extra or "").strip()
            if text:
                names.append(text)
        findings = _finding_hits(driver, db, uid, names)
    except Exception:
        logger.exception("companion cypher knowledge failed")

    return {"user_memory": memory, "knowledge_hits": knowledge, "finding_hits": findings}


def _query_rows(driver: Any, db: str, cypher: str) -> list[dict[str, Any]]:
    try:
        result = driver.execute_query(cypher, database_=db)
    except Exception:
        logger.exception("companion catalog query failed")
        return []
    return [row.data() for row in result.records]


def _finding_hits(driver: Any, db: str, user_id: str, names: list[str]) -> list[dict[str, Any]]:
    keywords = list(dict.fromkeys(item for item in names if item))[:12]
    if not user_id or not keywords:
        return []
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)-[:HAVE]->(f:Finding)
        WHERE toString(u.id) = $user_id
          AND any(name IN $names WHERE coalesce(f.raw_text, '') CONTAINS name)
        RETURN f.id AS id, f.raw_text AS raw_text, f.created_at AS created_at
        ORDER BY f.created_at DESC
        LIMIT 5
        """,
        parameters_={"user_id": user_id, "names": keywords},
        database_=db,
    )
    rows: list[dict[str, Any]] = []
    for rec in result.records:
        raw = str(rec.get("raw_text") or "").strip()
        rows.append(
            {
                "id": rec.get("id") or "",
                "raw_text": raw[:200],
                "created_at": rec.get("created_at") or "",
            }
        )
    return rows


def _lookup_node(driver: Any, db: str, *, label: str, node_id: Any, name: str) -> dict[str, Any] | None:
    if label == "ConsultBodyPart":
        cypher = """
        MATCH (n:ConsultBodyPart)
        WHERE ($node_id IS NOT NULL AND n.id = $node_id)
           OR ($name <> '' AND n.part_name = $name)
        RETURN 'ConsultBodyPart' AS label, n.id AS id, n.part_name AS name
        LIMIT 1
        """
    elif label == "ConsultDisease":
        cypher = """
        MATCH (n:ConsultDisease)
        WHERE ($node_id IS NOT NULL AND n.id = $node_id)
           OR ($name <> '' AND n.disease_name = $name)
        RETURN 'ConsultDisease' AS label, n.id AS id, n.disease_name AS name
        LIMIT 1
        """
    elif label == "ConsultPrimarySymptom":
        cypher = """
        MATCH (n:ConsultPrimarySymptom)
        WHERE ($node_id IS NOT NULL AND n.id = $node_id)
           OR ($name <> '' AND n.symptom_name = $name)
        RETURN 'ConsultPrimarySymptom' AS label, n.id AS id, n.symptom_name AS name
        LIMIT 1
        """
    else:
        return {"label": label, "id": node_id, "name": name} if name or node_id else None

    nid = node_id if isinstance(node_id, int) else None
    result = driver.execute_query(
        cypher,
        parameters_={"node_id": nid, "name": name},
        database_=db,
    )
    if not result.records:
        return None
    return result.records[0].data()
