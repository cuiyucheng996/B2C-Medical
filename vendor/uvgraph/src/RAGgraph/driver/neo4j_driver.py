"""8.2 创建 Driver 对象：Neo4j 连接池唯一入口。

Driver 不可变、线程安全；应用内复用单例，避免每次 API 请求重复握手。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from src.configs import NEO4J_CONFIG, app_settings

logger = logging.getLogger(__name__)


def _neo4j_auth() -> tuple[str, str]:
    password = app_settings.neo4j_password
    if not password:
        raise RuntimeError("NEO4J_PASSWORD 未配置，请在 .env 中填写")
    return NEO4J_CONFIG.get("user", "neo4j"), password


@lru_cache
def get_neo4j_driver() -> Driver:
    """返回进程内共享 Driver（首次调用时验证连通性）。"""
    uri = NEO4J_CONFIG.get("uri", "neo4j://localhost:7687")
    auth = _neo4j_auth()
    driver = GraphDatabase.driver(uri, auth=auth)
    driver.verify_connectivity()
    logger.info("Connection established: %s", uri)
    return driver


def create_neo4j_driver() -> Driver:
    """兼容旧调用：返回共享 Driver。注意不要用 `with` 关闭它。"""
    return get_neo4j_driver()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    driver = get_neo4j_driver()
    driver.verify_connectivity()
    print("Connection established.")
