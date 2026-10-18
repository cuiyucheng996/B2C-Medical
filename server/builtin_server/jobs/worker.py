"""后台消费关场入库与文件解析。"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from server.builtin_server.jobs.queue import blocking_pop, mark_job

logger = logging.getLogger(__name__)


def _session_from_snapshot(snap: dict[str, Any]):
    from session import CompanionSession

    session = CompanionSession(
        session_id=str(snap.get("session_id") or ""),
        user_id=str(snap.get("user_id") or snap.get("group_id") or ""),
        created_at=str(snap.get("created_at") or ""),
        user=list(snap.get("user") or []),
        last_preprocess=snap.get("last_preprocess"),
        last_report=snap.get("last_report"),
        last_file=snap.get("last_file"),
        slot_memory=dict(snap.get("slot_memory") or {}),
        persisted=bool(snap.get("persisted")),
        last_activity_at=str(snap.get("last_activity_at") or ""),
    )
    return session


def _handle(job: dict[str, Any]) -> None:
    job_id = str(job.get("id") or "")
    kind = str(job.get("kind") or "")
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    mark_job(job_id, status="running")
    if kind == "session_persist":
        from memory.graphiti_store import persist_session_findings

        result = persist_session_findings(_session_from_snapshot(payload.get("snapshot") or {}))
        mark_job(job_id, status="done", extra=json.dumps({"ingested": result.get("ingested")}, ensure_ascii=False))
        return
    if kind == "doc_ingest":
        from server.builtin_server.doc_ingest.logic import ingest_document

        path = Path(str(payload.get("path") or ""))
        result = ingest_document(
            filename=str(payload.get("filename") or path.name),
            data=path.read_bytes(),
            heading=payload.get("heading") if isinstance(payload.get("heading"), dict) else None,
        )
        mark_job(job_id, status="done" if result.get("ok") else "failed", extra=json.dumps(result, ensure_ascii=False)[:2000])
        return
    if kind == "chat_file":
        from algorithm.file_gate import ingest_user_file

        path = Path(str(payload.get("path") or ""))
        ingested = ingest_user_file(
            user_id=str(payload.get("user_id") or ""),
            filename=str(payload.get("filename") or path.name),
            data=path.read_bytes(),
        )
        mark_job(job_id, status="done", extra=json.dumps(ingested.to_dict(), ensure_ascii=False)[:2000])
        return
    mark_job(job_id, status="failed", error=f"unknown kind {kind}")


def run_forever() -> None:
    logger.info("heavy worker started")
    while True:
        try:
            job = blocking_pop(5)
        except Exception as exc:
            logger.warning("Redis 队列不可用，5 秒后重试: %s", exc)
            time.sleep(5)
            continue
        if not job:
            continue
        job_id = str(job.get("id") or "")
        try:
            _handle(job)
            logger.info("heavy job done %s", job_id)
        except Exception as exc:
            logger.exception("heavy job failed %s", job_id)
            if job_id:
                mark_job(job_id, status="failed", error=str(exc))
