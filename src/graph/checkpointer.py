"""陪聊 LangGraph checkpointer：仅运行中/中断续跑，非关场长期存储。"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from configs.paths_b2c import B2C_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "companion_checkpoint_enabled",
    "companion_graph_thread_id",
    "companion_invoke_config",
    "companion_recursion_limit",
    "get_companion_checkpointer",
]


def companion_checkpoint_enabled() -> bool:
    mode = (os.getenv("COMPANION_CHECKPOINTER") or "memory").strip().lower()
    return mode not in {"0", "false", "no", "off", "none", "disabled"}


def companion_graph_thread_id(base_turn: dict[str, Any] | None, *, explicit: str = "") -> str:
    tid = (explicit or "").strip()
    if tid:
        return tid
    turn = base_turn if isinstance(base_turn, dict) else {}
    sid = str(turn.get("session_id") or "").strip()
    if sid:
        return sid
    uid = str(turn.get("user_id") or "").strip()
    if uid:
        return f"user:{uid}"
    return "companion:anonymous"


def companion_recursion_limit() -> int:
    raw = (os.getenv("COMPANION_RECURSION_LIMIT") or "64").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 64
    return max(n, 25)


def companion_invoke_config(thread_id: str) -> dict[str, Any]:
    tid = (thread_id or "").strip() or "companion:anonymous"
    return {
        "configurable": {"thread_id": tid},
        "recursion_limit": companion_recursion_limit(),
    }


def _sqlite_path() -> Path:
    raw = (os.getenv("COMPANION_CHECKPOINT_SQLITE") or "").strip()
    if raw:
        return Path(raw)
    data_dir = B2C_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "companion_checkpoints.sqlite"


@lru_cache(maxsize=1)
def get_companion_checkpointer() -> Any | None:
    if not companion_checkpoint_enabled():
        logger.info("companion checkpointer: disabled (COMPANION_CHECKPOINTER=none)")
        return None
    mode = (os.getenv("COMPANION_CHECKPOINTER") or "memory").strip().lower()
    if mode in {"memory", "mem", "inmemory"}:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.info("companion checkpointer: InMemorySaver (process-local)")
        return InMemorySaver()
    if mode in {"sqlite", "disk", "file"}:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite 未安装，回退 InMemorySaver；"
                "本地调试 interrupt 可 pip install langgraph-checkpoint-sqlite"
            )
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()
        path = _sqlite_path()
        logger.info("companion checkpointer: AsyncSqliteSaver path=%s (dev/interrupt)", path)
        return AsyncSqliteSaver.from_conn_string(str(path))
    logger.warning("未知 COMPANION_CHECKPOINTER=%s，使用 InMemorySaver", mode)
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
