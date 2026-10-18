"""重活队列：Redis List，8002 只入队，server/worker 消费。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

QUEUE_KEY = "b2c:heavy:queue"
STATUS_PREFIX = "b2c:heavy:job:"


def async_heavy_enabled() -> bool:
    raw = (os.getenv("B2C_ASYNC_HEAVY") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _redis():
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from redis import Redis as SyncRedis
    from src.configs import REDIS_CONFIG

    return SyncRedis(
        host=REDIS_CONFIG["host"],
        port=int(REDIS_CONFIG["port"]),
        db=int(REDIS_CONFIG.get("db") or 0),
        password=REDIS_CONFIG.get("password") or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def enqueue(kind: str, payload: dict[str, Any]) -> str:
    job_id = uuid4().hex[:16]
    body = {"id": job_id, "kind": kind, "payload": payload}
    client = _redis()
    client.hset(
        STATUS_PREFIX + job_id,
        mapping={"status": "queued", "kind": kind, "error": ""},
    )
    client.expire(STATUS_PREFIX + job_id, 7 * 24 * 3600)
    client.lpush(QUEUE_KEY, json.dumps(body, ensure_ascii=False))
    logger.info("enqueued heavy job %s kind=%s", job_id, kind)
    return job_id


def job_status(job_id: str) -> dict[str, str]:
    data = _redis().hgetall(STATUS_PREFIX + (job_id or "").strip())
    return data if isinstance(data, dict) else {}


def mark_job(job_id: str, *, status: str, error: str = "", extra: str = "") -> None:
    client = _redis()
    client.hset(
        STATUS_PREFIX + job_id,
        mapping={"status": status, "error": (error or "")[:500], "extra": (extra or "")[:2000]},
    )


def blocking_pop(timeout: int = 5) -> dict[str, Any] | None:
    item = _redis().brpop(QUEUE_KEY, timeout=timeout)
    if not item:
        return None
    raw = item[1] if isinstance(item, (list, tuple)) else ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
