"""C 端测试账号：写在 Neo4j UserInfo 上，login_name 方便登录。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from configs.paths_b2c import ensure_uvgraph_root

__all__ = ["create_user", "get_user", "list_users", "login_user", "resolve_user_ref"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _pin_hash(pin: str) -> str:
    raw = (pin or "").strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _driver_db():
    ensure_uvgraph_root()
    from src.configs import NEO4J_CONFIG
    from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver

    return get_neo4j_driver(), str(NEO4J_CONFIG.get("database") or "neo4j")


def _row(rec: dict[str, Any]) -> dict[str, str]:
    return {
        "user_id": str(rec.get("id") or ""),
        "login_name": str(rec.get("login_name") or ""),
        "display_name": str(rec.get("display_name") or rec.get("login_name") or rec.get("id") or ""),
        "source": str(rec.get("source") or ""),
        "has_pin": bool(rec.get("has_pin")),
    }


def list_users(*, limit: int = 80) -> list[dict[str, str]]:
    driver, db = _driver_db()
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)
        RETURN toString(u.id) AS id,
               coalesce(u.login_name, '') AS login_name,
               coalesce(u.display_name, '') AS display_name,
               coalesce(u.source, '') AS source,
               coalesce(u.pin_hash, '') <> '' AS has_pin
        ORDER BY coalesce(u.login_name, toString(u.id))
        LIMIT $limit
        """,
        parameters_={"limit": int(limit)},
        database_=db,
    )
    return [_row(rec.data()) for rec in result.records]


def get_user(user_id: str) -> dict[str, str] | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    driver, db = _driver_db()
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE toString(u.id) = $user_id
        RETURN toString(u.id) AS id,
               coalesce(u.login_name, '') AS login_name,
               coalesce(u.display_name, '') AS display_name,
               coalesce(u.source, '') AS source,
               coalesce(u.pin_hash, '') <> '' AS has_pin
        LIMIT 1
        """,
        parameters_={"user_id": uid},
        database_=db,
    )
    if not result.records:
        return None
    return _row(result.records[0].data())


def create_user(*, login_name: str, display_name: str = "", pin: str = "") -> dict[str, Any]:
    name = (login_name or "").strip()
    if not name:
        return {"ok": False, "reason": "login_name_required"}
    driver, db = _driver_db()
    existed = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE u.login_name = $login_name
        RETURN toString(u.id) AS id
        LIMIT 1
        """,
        parameters_={"login_name": name},
        database_=db,
    )
    if existed.records:
        return {"ok": False, "reason": "login_name_taken"}
    user_id = str(uuid4())
    shown = (display_name or "").strip() or name
    driver.execute_query(
        """
        CREATE (u:UserInfo {
          id: $id,
          login_name: $login_name,
          display_name: $display_name,
          pin_hash: $pin_hash,
          source: 'b2c_register',
          create_time: $created_at
        })
        """,
        parameters_={
            "id": user_id,
            "login_name": name,
            "display_name": shown,
            "pin_hash": _pin_hash(pin),
            "created_at": _now(),
        },
        database_=db,
    )
    user = get_user(user_id)
    return {"ok": True, "user": user}


def resolve_user_ref(raw: str) -> str:
    """登录名或 id 都归一成 UserInfo.id；库里没有则原样返回。"""
    name = (raw or "").strip()
    if not name:
        return ""
    try:
        driver, db = _driver_db()
        result = driver.execute_query(
            """
            MATCH (u:UserInfo)
            WHERE u.login_name = $name OR toString(u.id) = $name
            RETURN toString(u.id) AS id,
                   coalesce(u.login_name, '') AS login_name
            LIMIT 1
            """,
            parameters_={"name": name},
            database_=db,
        )
    except Exception:
        return name
    if not result.records:
        return name
    rec = result.records[0].data()
    return str(rec.get("id") or rec.get("login_name") or name)


def login_user(*, login_name: str, pin: str = "") -> dict[str, Any]:
    name = (login_name or "").strip()
    if not name:
        return {"ok": False, "reason": "login_name_required"}
    driver, db = _driver_db()
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE u.login_name = $name OR toString(u.id) = $name
        RETURN toString(u.id) AS id,
               coalesce(u.login_name, '') AS login_name,
               coalesce(u.display_name, '') AS display_name,
               coalesce(u.source, '') AS source,
               coalesce(u.pin_hash, '') AS pin_hash
        LIMIT 1
        """,
        parameters_={"name": name},
        database_=db,
    )
    if not result.records:
        return {"ok": False, "reason": "not_found"}
    rec = result.records[0].data()
    stored = str(rec.get("pin_hash") or "")
    if stored and stored != _pin_hash(pin):
        return {"ok": False, "reason": "bad_pin"}
    return {
        "ok": True,
        "user": {
            "user_id": str(rec.get("id") or ""),
            "login_name": str(rec.get("login_name") or ""),
            "display_name": str(rec.get("display_name") or rec.get("login_name") or rec.get("id") or ""),
            "source": str(rec.get("source") or ""),
            "has_pin": bool(stored),
        },
    }
