"""陪聊 session 释放：入库、草稿、从 SessionStore 摘除。"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from memory.graphiti_store import persist_session_findings

if TYPE_CHECKING:
    from session import CompanionSession, SessionStore

__all__ = [
    "chat_draft_patch_from_session",
    "release_companion_session",
    "session_idle_seconds",
]


def session_idle_seconds() -> int:
    import os

    raw = (os.getenv("B2C_SESSION_IDLE_SECONDS") or "7200").strip()
    try:
        sec = int(raw)
    except ValueError:
        sec = 7200
    return max(60, sec)


def chat_draft_patch_from_session(session: CompanionSession) -> dict[str, Any]:
    from datetime import datetime, timezone

    messages: list[dict[str, str]] = []
    for item in session.user:
        role_type = str(item.get("role_type") or item.get("role") or "user")
        if role_type not in {"user", "assistant", "system"}:
            role_type = "user"
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        messages.append(
            {
                "role_type": role_type,
                "role": str(item.get("role") or role_type),
                "content": content,
            }
        )
    return {
        "chat": {
            "session_id": session.session_id,
            "messages": messages,
            "released_at": datetime.now(timezone.utc).isoformat(),
        }
    }


def release_companion_session(
    session: CompanionSession,
    store: SessionStore,
    *,
    reason: str = "explicit",
) -> dict[str, Any]:
    from configs.paths_b2c import import_b2c_algorithm

    release_session_runtime_state = import_b2c_algorithm(
        "session_memory"
    ).release_session_runtime_state
    merge_user_ui_draft = import_b2c_algorithm("user_ui_draft").merge_user_ui_draft

    from server.builtin_server.jobs.queue import async_heavy_enabled, enqueue

    if async_heavy_enabled():
        job_id = enqueue(
            "session_persist",
            {"snapshot": session.to_dict(), "reason": reason},
        )
        ingest = {"ok": True, "deferred": True, "job_id": job_id}
    else:
        ingest = persist_session_findings(session)
    patch = chat_draft_patch_from_session(session)
    patch["chat"]["release_reason"] = reason
    merge_user_ui_draft(session.user_id, patch)
    release_session_runtime_state(
        session.user_id,
        session.session_id,
        clear_slots=True,
        clear_checkpoint=True,
    )
    store.drop(session)
    return {"session": session.to_dict(), "ingest": ingest}
