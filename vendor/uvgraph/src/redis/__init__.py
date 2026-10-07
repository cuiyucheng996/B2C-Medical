"""Redis 服务层：服务对象建立 + 短期对话存储。

- ``client.py``：async 单例连接（配置来自 ``src.configs.REDIS_CONFIG``）
- ``conversation.py``：短期对话 API（user+assistant 配对；会话结束再释放；上下文最多 20 对）
"""

from src.redis.client import get_redis_client
from src.redis.conversation import (
    SHORT_MAX_ITEMS,
    SHORT_TTL_SECONDS,
    append_exchange,
    clear_conversation,
    clear_conversation_sync,
    context_messages_sync,
    get_context_exchanges,
    get_context_exchanges_sync,
    get_exchange_count,
    get_recent_exchanges,
    get_recent_exchanges_sync,
    short_conversation_key,
    sync_conversation_from_messages_sync,
)

__all__ = [
    "SHORT_MAX_ITEMS",
    "SHORT_TTL_SECONDS",
    "append_exchange",
    "clear_conversation",
    "clear_conversation_sync",
    "context_messages_sync",
    "get_context_exchanges",
    "get_context_exchanges_sync",
    "get_exchange_count",
    "get_recent_exchanges",
    "get_recent_exchanges_sync",
    "get_redis_client",
    "short_conversation_key",
    "sync_conversation_from_messages_sync",
]
