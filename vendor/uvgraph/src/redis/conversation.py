"""医疗项目短期对话存储：按 user_id 存对话组，会话结束再释放。

- key 约定：``{namespace}:{user_id}``（问诊图把 thread_id 当作这个后缀传入）
- Redis List，元素为 JSON 字符串
- 每个元素是**一组对话**（user + assistant 配对）：
  ``{"exchange": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}], "ts": ...}``
- 会话进行中不主动过期；List 最多保留 ``SHORT_MAX_ITEMS`` 组（LTRIM）
- 进入 LLM 上下文时最多取 ``SHORT_MAX_ITEMS``（默认 20）对
- 空间释放：``finalize_session`` / ``clear_conversation`` 在会话结束时删除 key
- 可选 ``SHORT_TTL_SECONDS>0`` 仅作异常中断兜底，正常不依赖 TTL
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from redis import Redis as SyncRedis

from src.configs import REDIS_CONFIG
from src.redis.client import get_redis_client

logger = logging.getLogger(__name__)

# 短期对话默认值（可被 config.yaml 的 redis 段覆盖）
SHORT_NS = str(REDIS_CONFIG.get("short_conversation_namespace") or "conversation:short").strip()
# 0 = 不设 TTL，由会话结束时 clear；>0 仅异常中断兜底
SHORT_TTL_SECONDS = int(REDIS_CONFIG.get("short_conversation_ttl_seconds") or 0)
# 进入上下文 / List 保留的最大对话组数
SHORT_MAX_ITEMS = int(REDIS_CONFIG.get("short_conversation_max_items") or 20)

__all__ = [
    "SHORT_MAX_ITEMS",
    "SHORT_TTL_SECONDS",
    "append_exchange",
    "clear_conversation",
    "clear_conversation_sync",
    "context_messages_b2c_session_sync",
    "context_messages_sync",
    "get_context_exchanges",
    "get_context_exchanges_sync",
    "get_exchange_count",
    "get_recent_exchanges",
    "get_recent_exchanges_sync",
    "short_conversation_key",
    "sync_b2c_session_conversation_sync",
    "sync_conversation_from_messages_sync",
]


def short_conversation_key(user_id: str) -> str:
    """生成短期对话的 Redis key。"""
    return f"{SHORT_NS}:{(user_id or '').strip()}"


def _sync_redis() -> SyncRedis:
    return SyncRedis(
        host=REDIS_CONFIG["host"],
        port=int(REDIS_CONFIG["port"]),
        db=int(REDIS_CONFIG.get("db") or 0),
        password=REDIS_CONFIG.get("password") or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


async def append_exchange(
    user_id: str,
    user_message: str,
    assistant_message: str,
    **extra: Any,
) -> int:
    """向会话追加**一组**对话（user + assistant），返回当前组数。

    会话中不设过期；仅 LTRIM 保留最近 ``SHORT_MAX_ITEMS`` 组。
    ``SHORT_TTL_SECONDS>0`` 时才写 TTL（异常兜底）。
    """
    redis_client = await get_redis_client()
    key = short_conversation_key(user_id)
    now = datetime.now(timezone.utc).isoformat()
    exchange = json.dumps(
        {
            "exchange": [
                {"role": "user", "content": user_message, "ts": now},
                {"role": "assistant", "content": assistant_message, "ts": now},
            ],
            "ts": now,
            **extra,
        },
        ensure_ascii=False,
    )
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(key, exchange)
        pipe.ltrim(key, -SHORT_MAX_ITEMS, -1)  # 只保留最近 N 组（上下文上限）
        if SHORT_TTL_SECONDS > 0:
            pipe.expire(key, SHORT_TTL_SECONDS)  # 仅异常中断兜底
        await pipe.execute()
    return int(await redis_client.llen(key))


def _role_content(message: Any) -> tuple[str, str] | None:
    if isinstance(message, dict):
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
    else:
        msg_type = str(getattr(message, "type", "") or "").strip().lower()
        if msg_type == "human":
            role = "user"
        elif msg_type == "ai":
            role = "assistant"
        else:
            role = msg_type
        content = getattr(message, "content", "")
    if role in ("human", "user"):
        role = "user"
    elif role in ("ai", "assistant"):
        role = "assistant"
    else:
        return None
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    return role, content


def _messages_to_exchanges(messages: list[Any]) -> list[tuple[str, str]]:
    """把 checkpoint messages 收成 user+assistant 对；未配对的 user 暂记空回复。"""
    pairs: list[tuple[str, str]] = []
    pending_user = ""
    for message in messages or []:
        parsed = _role_content(message)
        if parsed is None:
            continue
        role, content = parsed
        if role == "user":
            if pending_user:
                pairs.append((pending_user, ""))
            pending_user = content
        else:
            pairs.append((pending_user, content))
            pending_user = ""
    if pending_user:
        pairs.append((pending_user, ""))
    return pairs


def _dump_exchange(user_message: str, assistant_message: str, extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "exchange": [
            {"role": "user", "content": user_message, "ts": now},
            {"role": "assistant", "content": assistant_message, "ts": now},
        ],
        "ts": now,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def sync_b2c_session_conversation_sync(session_id: str, messages: list[Any]) -> int:
    """B2C 陪聊：按 session_id 存本场全部 user/assistant 对，不 LTRIM。"""
    sid = (session_id or "").strip()
    if not sid:
        return 0
    window = _messages_to_exchanges(messages)
    key = short_conversation_key(sid)
    client = _sync_redis()
    try:
        pipe = client.pipeline(transaction=True)
        pipe.delete(key)
        for user_message, assistant_message in window:
            pipe.rpush(key, _dump_exchange(user_message, assistant_message))
        if window and SHORT_TTL_SECONDS > 0:
            pipe.expire(key, SHORT_TTL_SECONDS)
        pipe.execute()
        return len(window)
    finally:
        client.close()


def sync_conversation_from_messages_sync(user_id: str, messages: list[Any]) -> int:
    """用 checkpoint 消息重建滑动窗口（跨实例共享，只留最近 SHORT_MAX_ITEMS 组）。"""
    sid = (user_id or "").strip()
    if not sid:
        return 0
    window = _messages_to_exchanges(messages)[-SHORT_MAX_ITEMS:]
    key = short_conversation_key(sid)
    client = _sync_redis()
    try:
        pipe = client.pipeline(transaction=True)
        pipe.delete(key)
        for user_message, assistant_message in window:
            pipe.rpush(key, _dump_exchange(user_message, assistant_message))
        if window and SHORT_TTL_SECONDS > 0:
            pipe.expire(key, SHORT_TTL_SECONDS)
        pipe.execute()
        return len(window)
    finally:
        client.close()


def get_recent_exchanges_sync(
    user_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    sid = (user_id or "").strip()
    if not sid:
        return []
    key = short_conversation_key(sid)
    client = _sync_redis()
    try:
        if limit is None:
            raw_list = client.lrange(key, 0, -1)
        else:
            raw_list = client.lrange(key, -max(int(limit), 0), -1)
        return _parse_exchanges(raw_list)
    finally:
        client.close()


def get_context_exchanges_sync(user_id: str) -> list[dict[str, Any]]:
    return get_recent_exchanges_sync(user_id, limit=SHORT_MAX_ITEMS)


def context_messages_sync(user_id: str) -> list[dict[str, str]]:
    """滑动窗口展平为 ``{role, content}``，供 LLM messages 使用。"""
    rows: list[dict[str, str]] = []
    for item in get_context_exchanges_sync(user_id):
        for turn in item.get("exchange") or []:
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                rows.append({"role": role, "content": content})
    return rows


def context_messages_b2c_session_sync(session_id: str) -> list[dict[str, str]]:
    """B2C：本场 session 全部对话展平，不截断组数。"""
    rows: list[dict[str, str]] = []
    for item in get_recent_exchanges_sync(session_id, limit=None):
        for turn in item.get("exchange") or []:
            role = str(turn.get("role") or "").strip()
            content = str(turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                rows.append({"role": role, "content": content})
    return rows


def _parse_exchanges(raw_list: list[Any]) -> list[dict[str, Any]]:
    exchanges: list[dict[str, Any]] = []
    for raw in raw_list:
        try:
            exchanges.append(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            logger.warning("skip invalid conversation exchange: %r", raw)
    return exchanges


async def get_recent_exchanges(
    user_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """取会话最近 ``limit`` 组对话，按时间从旧到新；``limit=None`` 取全部（仍受 List 上限约束）。"""
    redis_client = await get_redis_client()
    key = short_conversation_key(user_id)
    if limit is None:
        raw_list = await redis_client.lrange(key, 0, -1)
    else:
        raw_list = await redis_client.lrange(key, -max(int(limit), 0), -1)
    return _parse_exchanges(raw_list)


async def get_context_exchanges(user_id: str) -> list[dict[str, Any]]:
    """进入 LLM 上下文用的对话组：最多 ``SHORT_MAX_ITEMS``（默认 20）对。"""
    return await get_recent_exchanges(user_id, limit=SHORT_MAX_ITEMS)


async def get_exchange_count(user_id: str) -> int:
    """会话当前对话组数（键不存在返回 0）。"""
    redis_client = await get_redis_client()
    return int(await redis_client.llen(short_conversation_key(user_id)))


async def clear_conversation(user_id: str) -> bool:
    """会话结束时释放短期对话缓存，返回是否删除了键。"""
    redis_client = await get_redis_client()
    return bool(await redis_client.delete(short_conversation_key(user_id)))


def clear_conversation_sync(user_id: str) -> bool:
    """同步释放（供 ``finalize_session`` 等同步路径调用）。"""
    client = _sync_redis()
    try:
        return bool(client.delete(short_conversation_key(user_id)))
    finally:
        client.close()


if __name__ == "__main__":
    import asyncio
    import sys

    async def _main() -> None:
        user_id = (sys.argv[1] if len(sys.argv) > 1 else "demo-session").strip()
        await clear_conversation(user_id)
        await append_exchange(user_id, "你好", "你好，请问哪里不舒服？")
        await append_exchange(user_id, "我头疼", "头痛多久了？")
        print(f"count={await get_exchange_count(user_id)}")
        print("context:", await get_context_exchanges(user_id))
        await clear_conversation(user_id)
        print("cleared, count=", await get_exchange_count(user_id))

    asyncio.run(_main())
