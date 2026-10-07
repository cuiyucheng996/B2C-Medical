"""通用 Markdown 表格 → MySQL 入库。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from src.configs import MYSQL_CONFIG, MYSQL_INGEST_DATABASE
from src.knowledge.md_table import parse_markdown_table

_IDENT_RE = re.compile(r"^[\w]+$", re.UNICODE)


def resolve_ingest_database(database: str | None = None) -> str:
    """解析入库库名：CLI 覆盖 > config.mysql.ingest_database > mysql.database。"""
    name = (database or MYSQL_INGEST_DATABASE or MYSQL_CONFIG["database"]).strip()
    if not name:
        msg = "ingest database name is empty"
        raise ValueError(msg)
    return name


def table_name_from_markdown(path: Path) -> str:
    """表名 = md 文件名（不含扩展名），仅保留字母数字下划线。"""
    stem = path.stem.strip().lower()
    table_name = re.sub(r"[^\w]+", "_", stem, flags=re.UNICODE).strip("_")
    if not table_name:
        msg = f"cannot derive table name from {path}"
        raise ValueError(msg)
    return table_name


def quote_ident(name: str) -> str:
    if not name or not _IDENT_RE.match(name):
        msg = f"invalid sql identifier: {name!r}"
        raise ValueError(msg)
    return f"`{name.replace('`', '``')}`"


def mysql_base_config() -> dict[str, Any]:
    return {
        "host": MYSQL_CONFIG["host"],
        "user": MYSQL_CONFIG["user"],
        "password": MYSQL_CONFIG["password"],
        "charset": MYSQL_CONFIG["charset"],
        "port": MYSQL_CONFIG["port"],
    }


def ensure_database(database: str) -> None:
    db_name = quote_ident(database)
    with pymysql.connect(**mysql_base_config()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {db_name} "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()


def _build_create_table_sql(table_name: str, columns: list[str]) -> str:
    """建表：首列固定为自增 id，其后为 md 表头列。"""
    quoted_table = quote_ident(table_name)
    column_defs = ",\n  ".join(
        f"{quote_ident(column)} TEXT COMMENT {column!r}" for column in columns
    )
    return f"""
CREATE TABLE IF NOT EXISTS {quoted_table} (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT '主键',
  {column_defs},
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()


def _build_insert_sql(table_name: str, columns: list[str]) -> str:
    quoted_table = quote_ident(table_name)
    quoted_columns = [quote_ident(column) for column in columns]
    placeholders = ", ".join(f"%({column})s" for column in columns)
    insert_cols = ", ".join(quoted_columns)
    return f"""
INSERT INTO {quoted_table} ({insert_cols})
VALUES ({placeholders});
""".strip()


def ingest_markdown_mysql(
    source_path: Path,
    *,
    database: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """将任意 markdown 表格写入 MySQL；库名可配置，表名取自文件名。"""
    md_path = source_path.resolve()
    db_name = resolve_ingest_database(database)
    table_name = table_name_from_markdown(md_path)
    rows = parse_markdown_table(md_path)
    columns = list(rows[0].keys())

    ensure_database(db_name)

    conn_config = mysql_base_config()
    conn_config["database"] = db_name

    create_sql = _build_create_table_sql(table_name, columns)
    insert_sql = _build_insert_sql(table_name, columns)
    quoted_table = quote_ident(table_name)

    with pymysql.connect(**conn_config) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(create_sql)
            if reset:
                cursor.execute(f"TRUNCATE TABLE {quoted_table}")
            if rows:
                cursor.executemany(insert_sql, rows)
            cursor.execute(f"SELECT COUNT(*) AS cnt FROM {quoted_table}")
            total = cursor.fetchone()["cnt"]
        conn.commit()

    return {
        "source": str(md_path),
        "database": db_name,
        "table": table_name,
        "columns": ["id", *columns],
        "inserted_rows": len(rows),
        "total_rows": total,
    }
