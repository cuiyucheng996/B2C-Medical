"""MySQL → Neo4j 同步连接配置。"""

from __future__ import annotations

import os
from typing import Any

from src.configs.config_loader import MYSQL_CONFIG


def get_mysql_sync_config() -> dict[str, Any]:
    """同步库连接：优先 mysql.sync_* / 环境变量，密码可用 MYSQL_SYNC_PASSWORD。"""
    sync_password = os.getenv("MYSQL_SYNC_PASSWORD", "").strip()
    return {
        "host": str(MYSQL_CONFIG.get("sync_host") or MYSQL_CONFIG["host"]).strip(),
        "port": int(MYSQL_CONFIG.get("sync_port") or MYSQL_CONFIG["port"]),
        "user": str(MYSQL_CONFIG.get("sync_user") or MYSQL_CONFIG["user"]).strip(),
        "password": sync_password or MYSQL_CONFIG["password"],
        "database": str(MYSQL_CONFIG.get("sync_database") or "ai_person").strip(),
        "charset": MYSQL_CONFIG.get("charset", "utf8mb4"),
    }
