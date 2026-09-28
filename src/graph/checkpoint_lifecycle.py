"""Checkpoint 生命周期：不充当长期记忆，关场或单轮结束后释放 thread。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "checkpoint_retention_mode",
    "maybe_release_after_run",
    "release_companion_thread",
    "release_companion_thread_sync",
]


def checkpoint_retention_mode() -> str:
    import os

    # run：每轮图跑完即删（默认，对齐「一问一图」）
    # session：仅关场删，便于同 session 内 interrupt 续跑（单进程）
    return (os.getenv("COMPANION_CHECKPOINT_RETENTION") or "run").strip().lower()


def _graph_with_checkpointer() -> Any | None:
    from graph.checkpointer import companion_checkpoint_enabled

    if not companion_checkpoint_enabled():
        return None
    from graph.companion import build_companion_graph

    graph = build_companion_graph()
    if not getattr(graph, "checkpointer", None):
        return None
    return graph


async def release_companion_thread(thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    graph = _graph_with_checkpointer()
    if graph is None:
        return
    from graph.checkpointer import companion_invoke_config

    cp = graph.checkpointer
    if cp is None:
        return
    try:
        if hasattr(cp, "adelete_thread"):
            await cp.adelete_thread(tid)
        elif hasattr(cp, "delete_thread"):
            cp.delete_thread(tid)
        else:
            logger.debug("checkpointer lacks delete_thread thread_id=%s", tid)
    except Exception:
        logger.warning("release companion checkpoint failed thread_id=%s", tid, exc_info=True)


def release_companion_thread_sync(thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    graph = _graph_with_checkpointer()
    if graph is None:
        return
    from graph.checkpointer import companion_invoke_config

    cp = graph.checkpointer
    if cp is None:
        return
    try:
        if hasattr(cp, "delete_thread"):
            cp.delete_thread(tid)
        else:
            logger.debug("checkpointer lacks delete_thread thread_id=%s", tid)
    except Exception:
        logger.warning("release companion checkpoint failed thread_id=%s", tid, exc_info=True)


async def maybe_release_after_run(thread_id: str) -> None:
    if checkpoint_retention_mode() != "run":
        return
    await release_companion_thread(thread_id)


def release_on_session_end(session_id: str) -> None:
    """关场：与 clear_b2c_short_memory 同级，删掉 LangGraph thread。"""
    sid = (session_id or "").strip()
    if not sid:
        return
    try:
        from algorithm.hitl_checkpoint import clear_session_hitl

        clear_session_hitl(sid)
    except Exception:
        logger.warning("release hitl checkpoint failed session_id=%s", sid, exc_info=True)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        release_companion_thread_sync(sid)
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(release_companion_thread(sid))
    except Exception:
        release_companion_thread_sync(sid)
