"""MySQL 表行 → 向量化 → Chroma 持久化目录。"""

from __future__ import annotations

import logging
from typing import Any

from src.embeddings import get_embeddings
from src.knowledge.chroma_store import chroma_persist_directory, insert_chroma_rows
from src.knowledge.md_mysql_ingest import resolve_ingest_database
from src.knowledge.mysql_milvus_ingest import (
    build_milvus_record,
    load_mysql_rows,
    resolve_vector_column,
)

logger = logging.getLogger(__name__)

__all__ = ["ingest_mysql_to_chroma"]


def ingest_mysql_to_chroma(
    table: str,
    *,
    database: str | None = None,
    collection_name: str | None = None,
    vector_column: str | None = None,
    reset: bool = False,
) -> dict[str, Any]:
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
    embed_column = resolve_vector_column(table, vector_column)
    chroma_rows = [
        {
            "source_table": table,
            "disease": record["disease"],
            "text": record["text"],
            "vector_text": record["vector_text"],
        }
        for record in records
    ]
    inserted = insert_chroma_rows(collection, chroma_rows, reset=reset, id_prefix=table)
    dim = len(get_embeddings().embed_query(chroma_rows[0]["vector_text"]))
    return {
        "database": db_name,
        "table": table,
        "collection": collection,
        "vector_column": embed_column or "all_columns",
        "inserted_rows": inserted,
        "vector_dim": dim,
        "persist_directory": str(chroma_persist_directory()),
    }
