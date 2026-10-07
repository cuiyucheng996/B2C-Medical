"""ai_persion 问诊表 × Milvus 疾病/症状：抽主症、补部位/证型、剩余写入辅症。"""

from __future__ import annotations

import json
import logging
from typing import Any

import pymysql
from pymilvus import MilvusClient
from pymysql.cursors import DictCursor

from src.configs import MILVUS_CONFIG, MYSQL_CONFIG
from src.configs.mysql_sync import get_mysql_sync_config
from src.configs.proxy import ensure_local_no_proxy
from src.knowledge.mysql_milvus_ingest import ensure_milvus_collection

logger = logging.getLogger(__name__)

__all__ = ["sync_consult_milvus_collections"]

COLLECTION_BODY_PART = "consult_body_part"
COLLECTION_SYNDROME = "syndrome_type"
COLLECTION_PRIMARY_SYMPTOM = "primary_symptom"
COLLECTION_AUXILIARY_SYMPTOM = "auxiliary_symptom"


def _mysql_conn():
    cfg = get_mysql_sync_config()
    if not str(cfg.get("user") or "").strip():
        cfg["user"] = MYSQL_CONFIG["user"]
    if not str(cfg.get("password") or "").strip():
        cfg["password"] = MYSQL_CONFIG["password"]
    return pymysql.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg.get("charset") or "utf8mb4",
        cursorclass=DictCursor,
    )


def _milvus_client() -> MilvusClient:
    ensure_local_no_proxy()
    kwargs = dict(MILVUS_CONFIG)
    if not str(kwargs.get("token") or "").strip():
        kwargs.pop("token", None)
    return MilvusClient(**kwargs)


def _norm(text: str) -> str:
    return (text or "").replace(" ", "").replace("　", "").strip()


def _parse_meta(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def _split_names(raw: str) -> list[str]:
    text = (raw or "").replace("，", "、").replace(",", "、")
    return [item.strip() for item in text.split("、") if item.strip()]


def _parse_json_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _split_names(text)
        return _parse_json_names(parsed)
    return []


def load_mysql_catalog() -> dict[str, Any]:
    with _mysql_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, part_code, part_name, summary FROM consult_body_part WHERE coalesce(status,1)=1"
            )
            parts = list(cursor.fetchall())
            cursor.execute(
                "SELECT id, disease_code, disease_name, symptom_type, description "
                "FROM consult_disease WHERE coalesce(status,1)=1"
            )
            diseases = list(cursor.fetchall())
            cursor.execute(
                "SELECT id, symptom_code, disease_code, symptom_name, description "
                "FROM consult_primary_symptom WHERE coalesce(status,1)=1"
            )
            primaries = list(cursor.fetchall())
            cursor.execute(
                "SELECT id, disease_name, syndrome_name, main_symptoms, supplementary_symptoms, status "
                "FROM syndrome_type"
            )
            syndromes = list(cursor.fetchall())
    return {
        "parts": parts,
        "diseases": diseases,
        "primaries": primaries,
        "syndromes": syndromes,
    }


def query_all(client: MilvusClient, collection: str, fields: list[str], batch: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        chunk = client.query(
            collection_name=collection,
            filter="",
            output_fields=fields,
            limit=batch,
            offset=offset,
        )
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += len(chunk)
    return rows


def milvus_symptom_names(row: dict[str, Any]) -> tuple[str, list[str]]:
    meta = _parse_meta(str(row.get("text") or ""))
    main = (meta.get("症状") or str(row.get("disease") or "")).strip()
    synonyms = _split_names(meta.get("通俗同义词") or "")
    names = [main, *synonyms] if main else list(synonyms)
    return main, [item for item in names if item]


def mysql_primary_name_set(catalog: dict[str, Any]) -> set[str]:
    names = {_norm(str(row.get("symptom_name") or "")) for row in catalog["primaries"]}
    names.discard("")
    return names


def _embed_isolated(texts: list[str]) -> list[list[float]]:
    """独立 python 进程做 embedding，避免 pymilvus + CUDA 同进程崩溃。"""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    worker = root / "scripts" / "embed_texts_worker.py"
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "texts.json"
        out_path = Path(tmp) / "vectors.json"
        in_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(worker), str(in_path), str(out_path)],
            check=True,
            cwd=str(root),
        )
        return json.loads(out_path.read_text(encoding="utf-8"))


def _insert_named_rows(
    client: MilvusClient,
    collection: str,
    items: list[dict[str, str]],
    *,
    reset: bool,
) -> int:
    if not items:
        logger.info("skip empty collection: %s", collection)
        return 0
    texts = [item["vector_text"] for item in items]
    logger.info("embedding %s rows for %s in subprocess", len(texts), collection)
    vectors = _embed_isolated(texts)
    dim = len(vectors[0])
    ensure_milvus_collection(client, collection, dim, reset=reset)
    payload = [
        {
            "source_table": item["source_table"],
            "disease": item["name"][:256],
            "text": item["text"][:4096],
            "vector": vector,
        }
        for item, vector in zip(items, vectors)
    ]
    inserted = 0
    for start in range(0, len(payload), 64):
        result = client.insert(collection_name=collection, data=payload[start : start + 64])
        inserted += int(result.get("insert_count", len(payload[start : start + 64])))
    logger.info("inserted %s into %s", inserted, collection)
    return inserted


def sync_consult_milvus_collections(*, reset: bool = True) -> dict[str, Any]:
    catalog = load_mysql_catalog()
    client = _milvus_client()
    if not client.has_collection("disease"):
        raise RuntimeError("Milvus 缺少 collection=disease")
    if not client.has_collection("symptoms"):
        raise RuntimeError("Milvus 缺少 collection=symptoms")

    milvus_diseases = query_all(client, "disease", ["id", "disease", "text"])
    milvus_symptoms = query_all(client, "symptoms", ["id", "disease", "text"])
    del client

    mysql_disease_names = {_norm(str(row.get("disease_name") or "")) for row in catalog["diseases"]}
    milvus_disease_names = {_norm(str(row.get("disease") or "")) for row in milvus_diseases}
    disease_overlap = sorted(mysql_disease_names & milvus_disease_names - {""})

    primary_set = mysql_primary_name_set(catalog)
    aux_rows: list[dict[str, Any]] = []
    for row in milvus_symptoms:
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

    client = _milvus_client()
    inserted = {
        COLLECTION_BODY_PART: _insert_named_rows(client, COLLECTION_BODY_PART, part_items, reset=reset),
        COLLECTION_SYNDROME: _insert_named_rows(client, COLLECTION_SYNDROME, syndrome_items, reset=reset),
        COLLECTION_PRIMARY_SYMPTOM: _insert_named_rows(
            client, COLLECTION_PRIMARY_SYMPTOM, primary_items, reset=reset
        ),
        COLLECTION_AUXILIARY_SYMPTOM: _insert_named_rows(
            client, COLLECTION_AUXILIARY_SYMPTOM, aux_items, reset=reset
        ),
    }

    report = {
        "mysql_database": get_mysql_sync_config()["database"],
        "mysql_counts": {
            "consult_body_part": len(catalog["parts"]),
            "consult_disease": len(catalog["diseases"]),
            "consult_primary_symptom": len(catalog["primaries"]),
            "syndrome_type": len(catalog["syndromes"]),
        },
        "milvus_existing": {
            "disease": len(milvus_diseases),
            "symptoms": len(milvus_symptoms),
        },
        "disease_name_overlap": len(disease_overlap),
        "disease_overlap_samples": disease_overlap[:20],
        "mysql_primary_names": len(primary_set),
        "syndrome_inactive": [
            {
                "id": row["id"],
                "disease_name": row.get("disease_name"),
                "syndrome_name": row.get("syndrome_name"),
                "status": row.get("status"),
            }
            for row in catalog["syndromes"]
            if int(row.get("status") or 1) != 1
        ],
        "split": {
            "primary_from_mysql": len(primary_items),
            "auxiliary_from_symptoms": len(aux_items),
            "full_symptoms_kept": "symptoms",
        },
        "inserted": inserted,
    }
    logger.info("consult milvus sync: %s", report)
    return report
