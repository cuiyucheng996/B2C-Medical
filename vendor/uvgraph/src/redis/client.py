"""Redis 服务对象：进程内单例 async client。

配置统一来自 ``src.configs.REDIS_CONFIG``（config.yaml 的 redis 段），
业务代码里不要散落 host/port/password。
"""

from __future__ import annotations

from redis.asyncio import Redis

from src.configs import REDIS_CONFIG

__all__ = ["get_redis_client"]

_client: Redis | None = None


async def get_redis_client() -> Redis:
    """返回进程内单例 Redis 连接（async，``decode_responses=True``）。"""
    global _client
    if _client is None:
        _client = Redis(
            host=REDIS_CONFIG["host"],
            port=int(REDIS_CONFIG["port"]),
            db=int(REDIS_CONFIG.get("db") or 0),
            password=REDIS_CONFIG.get("password") or None,
            decode_responses=True,
            # Redis 不可用时快速失败，避免请求无限挂起
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _client
