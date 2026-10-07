"""ai_persion 问诊表 → Chroma（逻辑同 consult_milvus_ingest，数据源改为 Chroma collection）。"""

from __future__ import annotations

import logging
from typing import Any

from src.configs.mysql_sync import get_mysql_sync_config
from src.knowledge.chroma_store import insert_chroma_rows, query_chroma_all
from src.knowledge.consult_milvus_ingest import (
    COLLECTION_AUXILIARY_SYMPTOM,
    COLLECTION_BODY_PART,
    COLLECTION_PRIMARY_SYMPTOM,
    COLLECTION_SYNDROME,
    _norm,
    _parse_json_names,
    load_mysql_catalog,
    milvus_symptom_names,
    mysql_primary_name_set,
)

logger = logging.getLogger(__name__)

__all__ = ["sync_consult_chroma_collections"]


def _insert_named_items(collection: str, items: list[dict[str, str]], *, reset: bool) -> int:
    rows = [
        {
            "source_table": item["source_table"],
            "disease": item["name"][:256],
            "name": item["name"][:256],
            "text": item["text"][:4096],
            "vector_text": item["vector_text"],
        }
        for item in items
        if item.get("vector_text")
    ]
    return insert_chroma_rows(collection, rows, reset=reset, id_prefix=collection)


def sync_consult_chroma_collections(*, reset: bool = True) -> dict[str, Any]:
    catalog = load_mysql_catalog()
    chroma_diseases = query_chroma_all("disease")
    chroma_symptoms = query_chroma_all("symptoms")
    if not chroma_diseases:
        raise RuntimeError("Chroma 缺少 collection=disease，请先 ingest_mysql_to_chroma")
    if not chroma_symptoms:
        raise RuntimeError("Chroma 缺少 collection=symptoms，请先入库 symptoms 表")

    mysql_disease_names = {_norm(str(row.get("disease_name") or "")) for row in catalog["diseases"]}
    chroma_disease_names = {_norm(str(row.get("disease") or "")) for row in chroma_diseases}
    disease_overlap = sorted(mysql_disease_names & chroma_disease_names - {""})

    primary_set = mysql_primary_name_set(catalog)
    aux_rows: list[dict[str, Any]] = []
    for row in chroma_symptoms:
        main, _ = milvus_symptom_names(row)
        if not main:
            continue
        if _norm(main) in primary_set:
            continue
        aux_rows.append(row)

    part_items = [
        {
            "source_table": "consult_body_part",
            "name": str(row["part_name"]),
            "vector_text": str(row["part_name"]),
            "text": (
                f"部位:{row['part_name']}\n"
                f"编码:{row.get('part_code') or ''}\n"
                f"mysql_id:{row['id']}\n"
                f"介绍:{row.get('summary') or ''}"
            ),
        }
        for row in catalog["parts"]
        if row.get("part_name")
    ]
    syndrome_items = [
        {
            "source_table": "syndrome_type",
            "name": str(row["syndrome_name"]),
            "vector_text": str(row["syndrome_name"]),
            "text": (
                f"证型:{row['syndrome_name']}\n"
                f"疾病:{row.get('disease_name') or ''}\n"
                f"mysql_id:{row['id']}\n"
                f"status:{row.get('status')}\n"
                f"主症:{','.join(_parse_json_names(row.get('main_symptoms')))}"
            ),
        }
        for row in catalog["syndromes"]
        if row.get("syndrome_name")
    ]

    def _symptom_item(row: dict[str, Any], source: str) -> dict[str, str]:
        main, _ = milvus_symptom_names(row)
        return {
            "source_table": source,
            "name": main,
            "vector_text": main,
            "text": str(row.get("text") or f"症状:{main}"),
        }

    primary_items = [
        {
            "source_table": "consult_primary_symptom",
            "name": str(row["symptom_name"]),
            "vector_text": str(row["symptom_name"]),
            "text": (
                f"症状:{row['symptom_name']}\n"
                f"编码:{row.get('symptom_code') or ''}\n"
                f"疾病编码:{row.get('disease_code') or ''}\n"
                f"mysql_id:{row['id']}\n"
                f"描述:{row.get('description') or ''}"
            ),
        }
        for row in catalog["primaries"]
        if row.get("symptom_name")
    ]
    aux_items = [_symptom_item(row, "auxiliary_symptom") for row in aux_rows if milvus_symptom_names(row)[0]]

    inserted = {
        COLLECTION_BODY_PART: _insert_named_items(COLLECTION_BODY_PART, part_items, reset=reset),
        COLLECTION_SYNDROME: _insert_named_items(COLLECTION_SYNDROME, syndrome_items, reset=reset),
        COLLECTION_PRIMARY_SYMPTOM: _insert_named_items(
            COLLECTION_PRIMARY_SYMPTOM, primary_items, reset=reset
        ),
        COLLECTION_AUXILIARY_SYMPTOM: _insert_named_items(
            COLLECTION_AUXILIARY_SYMPTOM, aux_items, reset=reset
        ),
    }

    report = {
        "mysql_database": get_mysql_sync_config()["database"],
        "chroma_existing": {"disease": len(chroma_diseases), "symptoms": len(chroma_symptoms)},
        "disease_name_overlap": len(disease_overlap),
        "inserted": inserted,
        "persist": "chroma",
    }
    logger.info("consult chroma sync: %s", report)
    return report
