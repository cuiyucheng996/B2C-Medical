"""MySQL 表行 → 向量化 → Milvus 入库。"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pymysql
from pymilvus import DataType, MilvusClient
from pymysql.cursors import DictCursor

from src.configs import MILVUS_CONFIG
from src.embeddings import get_embeddings
from src.knowledge.md_mysql_ingest import mysql_base_config, quote_ident, resolve_ingest_database

logger = logging.getLogger(__name__)

_SKIP_TEXT_COLUMNS = frozenset({"id", "created_at", "updated_at"})
_DISEASE_COLUMN_ALIASES = frozenset({"疾病", "disease"})


def format_mysql_row(row: dict[str, Any]) -> str:
    """将 MySQL 一行格式化为完整文本（入库 metadata，默认不参与向量化）。"""
    lines: list[str] = []
    for key, value in row.items():
        if key in _SKIP_TEXT_COLUMNS or value is None:
            continue
        if isinstance(value, Decimal):
            value = float(value)
        text = str(value).strip()
        if text:
            lines.append(f"{key}:{text}")
    return "\n".join(lines)


def extract_column_text(row: dict[str, Any], column: str) -> str:
    """读取指定列文本；列不存在或为空时返回空串。"""
    if column not in row:
        return ""
    value = row[column]
    if value is None:
        return ""
    if isinstance(value, Decimal):
        value = float(value)
    return str(value).strip()


def resolve_vector_column(table: str, vector_column: str | None) -> str | None:
    """解析用于 embedding 的列名；disease 表默认只用「疾病」列。"""
    if vector_column:
        return vector_column
    if table == "disease":
        return "疾病"
    return None


def build_milvus_record(
    row: dict[str, Any],
    *,
    table: str,
    vector_column: str | None,
) -> dict[str, str] | None:
    """组装单条记录：disease 单独列，vector 仅用指定列（默认疾病名）文本。"""
    full_text = format_mysql_row(row)
    if not full_text:
        return None

    embed_column = resolve_vector_column(table, vector_column)
    if embed_column:
        vector_text = extract_column_text(row, embed_column)
        if not vector_text:
            return None
        disease = vector_text if embed_column in _DISEASE_COLUMN_ALIASES else ""
    else:
        vector_text = full_text
        disease = ""

    return {
        "disease": disease,
        "text": full_text,
        "vector_text": vector_text,
    }


def load_mysql_rows(database: str, table: str) -> list[dict[str, Any]]:
    """读取 MySQL 表全部行（要求含自增 id）。"""
    quoted_table = quote_ident(table)
    conn_config = mysql_base_config()
    conn_config["database"] = database

    with pymysql.connect(**conn_config) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(f"SELECT * FROM {quoted_table} ORDER BY `id`")
            rows = list(cursor.fetchall())

    logger.info("loaded %s rows from %s.%s", len(rows), database, table)
    return rows


def ensure_milvus_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int,
    *,
    reset: bool = False,
) -> None:
    if client.has_collection(collection_name):
        if not reset:
            return
        client.drop_collection(collection_name)

    schema = MilvusClient.create_schema(auto_id=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="source_table", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="disease", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4096)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def ingest_mysql_to_milvus(
    table: str,
    *,
    database: str | None = None,
    collection_name: str | None = None,
    vector_column: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    """MySQL 表 → embedding → Milvus collection。

    vector_column 指定后仅该列文本做向量化；disease 表默认 ``疾病``。
    Milvus 中 ``disease`` 为单独字段；``text`` 保留整行 metadata。
    """
    db_name = resolve_ingest_database(database)
    collection = collection_name or table
    rows = load_mysql_rows(db_name, table)

    records: list[dict[str, str]] = []
    for row in rows:
        record = build_milvus_record(row, table=table, vector_column=vector_column)
        if record is not None:
            records.append(record)

    if not records:
        msg = f"no embeddable rows in {db_name}.{table}"
        raise ValueError(msg)

    vector_texts = [item["vector_text"] for item in records]
    embed_column = resolve_vector_column(table, vector_column)
    logger.info(
        "embedding %s rows from %s.%s using column=%s",
        len(vector_texts),
        db_name,
        table,
        embed_column or "all_columns",
    )

    embeddings = get_embeddings().embed_documents(vector_texts)
    dim = len(embeddings[0])
    logger.info("embedded %s docs, dim=%s", len(vector_texts), dim)

    client = MilvusClient(**MILVUS_CONFIG)
    ensure_milvus_collection(client, collection, dim, reset=reset)

    payload = [
        {
            "source_table": table,
            "disease": record["disease"],
            "text": record["text"],
            "vector": vector,
        }
        for record, vector in zip(records, embeddings)
    ]
    result = client.insert(collection_name=collection, data=payload)

    return {
        "database": db_name,
        "table": table,
        "collection": collection,
        "vector_column": embed_column or "all_columns",
        "inserted_rows": result.get("insert_count", len(payload)),
        "milvus_ids": result.get("ids", []),
        "vector_dim": dim,
    }
