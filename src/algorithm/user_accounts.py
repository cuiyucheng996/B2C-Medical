"""C 端测试账号：写在 Neo4j UserInfo 上，login_name 方便登录。"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from configs.paths_b2c import ensure_uvgraph_root

__all__ = ["create_user", "get_user", "list_users", "login_user", "resolve_user_ref"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


_PBKDF2_ROUNDS = 120_000
_SESSION_COOKIE = "b2c_session"
_SESSION_DAYS = 7


def _pin_hash(pin: str) -> str:
    raw = (pin or "").strip()
    if len(raw) < 4:
        return ""
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", raw.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS).hex()
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt}${digest}"


def _pin_ok(pin: str, stored: str) -> bool:
    raw = (pin or "").strip()
    saved = (stored or "").strip()
    if not saved or not raw:
        return False
    if saved.startswith("pbkdf2$"):
        try:
            _, rounds, salt, digest = saved.split("$", 3)
            check = hashlib.pbkdf2_hmac(
                "sha256", raw.encode("utf-8"), bytes.fromhex(salt), int(rounds)
            ).hex()
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(check, digest)
    legacy = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, saved)


def _session_secret() -> bytes:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs.config_loader import get_app_settings

    secret = (get_app_settings().n8n_webhook_secret or "").strip() or "b2c-dev-session"
    return secret.encode("utf-8")


def issue_session(user_id: str, role: str) -> str:
    exp = int((datetime.now(timezone.utc) + timedelta(days=_SESSION_DAYS)).timestamp())
    payload = f"{user_id}|{role}|{exp}"
    sig = hmac.new(_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def read_session(token: str) -> dict[str, str] | None:
    parts = (token or "").split("|")
    if len(parts) != 4:
        return None
    user_id, role, exp, sig = parts
    payload = f"{user_id}|{role}|{exp}"
    check = hmac.new(_session_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(check, sig):
        return None
    try:
        if int(exp) < int(datetime.now(timezone.utc).timestamp()):
            return None
    except ValueError:
        return None
    if role not in {"user", "admin"}:
        return None
    return {"user_id": user_id, "role": role}


def session_cookie_name() -> str:
    return _SESSION_COOKIE


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
        "role": str(rec.get("role") or "user"),
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
               coalesce(u.role, 'user') AS role,
               coalesce(u.pin_hash, '') <> '' AS has_pin
        ORDER BY coalesce(u.login_name, toString(u.id))
        LIMIT $limit
        """,
        parameters_={"limit": int(limit)},
        database_=db,
    )
    return [_row(rec.data()) for rec in result.records]


def ensure_one_admin() -> None:
    """旧账号没有 role 时，把最早的一条标成管理员，否则没人能进用户管理。"""
    driver, db = _driver_db()
    found = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE u.role = 'admin'
        RETURN toString(u.id) AS id
        LIMIT 1
        """,
        database_=db,
    )
    if found.records:
        return
    driver.execute_query(
        """
        MATCH (u:UserInfo)
        WITH u ORDER BY coalesce(u.create_time, '') ASC
        LIMIT 1
        SET u.role = 'admin'
        """,
        database_=db,
    )


def search_users(login_name: str = "", *, limit: int = 50) -> list[dict[str, str]]:
    """按登录名包含关系筛选。空关键字返回全部。"""
    driver, db = _driver_db()
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE $q = '' OR toLower(coalesce(u.login_name, '')) CONTAINS $q
        RETURN toString(u.id) AS id,
               coalesce(u.login_name, '') AS login_name,
               coalesce(u.display_name, '') AS display_name,
               coalesce(u.source, '') AS source,
               coalesce(u.role, 'user') AS role,
               coalesce(u.pin_hash, '') <> '' AS has_pin,
               coalesce(u.create_time, '') AS create_time
        ORDER BY coalesce(u.login_name, toString(u.id))
        LIMIT $limit
        """,
        parameters_={"q": (login_name or "").strip().lower(), "limit": int(limit)},
        database_=db,
    )
    rows = []
    for rec in result.records:
        row = _row(rec.data())
        row["create_time"] = str(rec.data().get("create_time") or "")
        rows.append(row)
    return rows


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
               coalesce(u.role, 'user') AS role,
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
    pin_hash = _pin_hash(pin)
    if not pin_hash:
        return {"ok": False, "reason": "pin_required"}
    user_id = str(uuid4())
    shown = (display_name or "").strip() or name
    count = driver.execute_query(
        "MATCH (u:UserInfo) RETURN count(u) AS n",
        database_=db,
    )
    role = "admin" if int(count.records[0].data().get("n") or 0) == 0 else "user"
    driver.execute_query(
        """
        CREATE (u:UserInfo {
          id: $id,
          login_name: $login_name,
          display_name: $display_name,
          pin_hash: $pin_hash,
          role: $role,
          source: 'b2c_register',
          create_time: $created_at
        })
        """,
        parameters_={
            "id": user_id,
            "login_name": name,
            "display_name": shown,
            "pin_hash": pin_hash,
            "role": role,
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
    ensure_one_admin()
    driver, db = _driver_db()
    result = driver.execute_query(
        """
        MATCH (u:UserInfo)
        WHERE u.login_name = $name OR toString(u.id) = $name
        RETURN toString(u.id) AS id,
               coalesce(u.login_name, '') AS login_name,
               coalesce(u.display_name, '') AS display_name,
               coalesce(u.source, '') AS source,
               coalesce(u.role, 'user') AS role,
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
    if not stored:
        return {"ok": False, "reason": "pin_required"}
    if not _pin_ok(pin, stored):
        return {"ok": False, "reason": "bad_pin"}
    role = str(rec.get("role") or "user")
    if role not in {"user", "admin"}:
        role = "user"
    user = {
        "user_id": str(rec.get("id") or ""),
        "login_name": str(rec.get("login_name") or ""),
        "display_name": str(rec.get("display_name") or rec.get("login_name") or rec.get("id") or ""),
        "source": str(rec.get("source") or ""),
        "role": role,
        "has_pin": True,
    }
    return {"ok": True, "user": user, "session": issue_session(user["user_id"], role)}
