"""每日陪伴小结：扫有长期记忆的用户，写入 CompanionDigest。"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["load_latest_digest", "mark_digest_read", "run_daily_digests"]

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_USER_MATCH = "toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id"


def _today() -> str:
    return datetime.now(_LOCAL_TZ).date().isoformat()


def _driver_db():
    ensure_uvgraph_root()
    from src.configs import NEO4J_CONFIG
    from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver

    return get_neo4j_driver(), str(NEO4J_CONFIG.get("database") or "neo4j")


def _compose(period: str, facts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for item in facts:
        text = str(item.get("fact") or item.get("name") or "").strip()
        if item.get("negated"):
            text = "已排除：" + str(item.get("name") or text)
        if not text or text.startswith("user("):
            continue
        key = text.replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        lines.append(text)
        if len(lines) >= 6:
            break
    if not lines:
        return ""
    bullets = "\n".join(f"- {line}" for line in lines)
    return (
        f"【每日陪伴小结 · {period}】\n"
        f"根据你留下的长期记忆，最近仍有效的记录：\n{bullets}\n"
        "这是定期归纳，不是体检报告，也不是诊断。有变化直接说。"
    )


def run_daily_digests(*, period: str | None = None) -> dict[str, Any]:
    """给每位有 CompanionFact 的用户生成当天小结；已有则跳过。"""
    day = period or _today()
    driver, db = _driver_db()
    listed = driver.execute_query(
        """
        MATCH (u:UserInfo)-[:STATED_FACT]->(:CompanionFact)
        RETURN DISTINCT toString(u.id) AS id
        """,
        database_=db,
    )
    written = 0
    skipped = 0
    users = [str(rec.data().get("id") or "") for rec in listed.records]
    ensure_uvgraph_root()
    from src.algorithm.user_memory import load_user_memory_card

    for uid in users:
        if not uid:
            continue
        key = hashlib.sha256(f"{uid}|daily|{day}".encode("utf-8")).hexdigest()
        existed = driver.execute_query(
            """
            MATCH (d:CompanionDigest {digest_key: $key})
            RETURN d.id AS id
            LIMIT 1
            """,
            parameters_={"key": key},
            database_=db,
        )
        if existed.records:
            skipped += 1
            continue
        card = load_user_memory_card(uid, limit=8)
        text = _compose(day, list(card.get("companion_facts") or []))
        if not text:
            skipped += 1
            continue
        driver.execute_query(
            f"""
            MATCH (u:UserInfo)
            WHERE {_USER_MATCH}
            WITH u LIMIT 1
            CREATE (d:CompanionDigest {{
              id: $digest_id,
              digest_key: $key,
              period: $period,
              kind: 'daily',
              raw_text: $text,
              created_at: $created_at,
              read: false
            }})
            MERGE (u)-[:HAS_DIGEST]->(d)
            """,
            parameters_={
                "user_id": uid,
                "digest_id": str(uuid4()),
                "key": key,
                "period": day,
                "text": text,
                "created_at": datetime.now(_LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            },
            database_=db,
        )
        written += 1
    logger.info("daily digest done period=%s written=%s skipped=%s", day, written, skipped)
    return {"ok": True, "period": day, "written": written, "skipped": skipped, "users": len(users)}


def load_latest_digest(user_id: str) -> dict[str, Any] | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    driver, db = _driver_db()
    result = driver.execute_query(
        f"""
        MATCH (u:UserInfo)-[:HAS_DIGEST]->(d:CompanionDigest)
        WHERE {_USER_MATCH}
        RETURN d.id AS id, d.raw_text AS raw_text, d.period AS period,
               d.created_at AS created_at, coalesce(d.read, false) AS read
        ORDER BY d.created_at DESC
        LIMIT 1
        """,
        parameters_={"user_id": uid},
        database_=db,
    )
    if not result.records:
        return None
    return result.records[0].data()


def mark_digest_read(digest_id: str) -> None:
    did = (digest_id or "").strip()
    if not did:
        return
    driver, db = _driver_db()
    driver.execute_query(
        """
        MATCH (d:CompanionDigest {id: $id})
        SET d.read = true
        """,
        parameters_={"id": did},
        database_=db,
    )
