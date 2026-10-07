"""Chroma 对齐/检索所需 collection 与 Milvus 时代一致。"""

from __future__ import annotations

from typing import Any

# MySQL 基础库（与 ingest_mysql_* 一致）
MYSQL_BASE_COLLECTIONS = ("disease", "symptoms")
# 问诊对齐（与 consult_*_ingest 一致）
CONSULT_ALIGN_COLLECTIONS = (
    "consult_body_part",
    "syndrome_type",
    "primary_symptom",
    "auxiliary_symptom",
)

REQUIRED_CHROMA_COLLECTIONS = MYSQL_BASE_COLLECTIONS + CONSULT_ALIGN_COLLECTIONS

__all__ = [
    "CONSULT_ALIGN_COLLECTIONS",
    "MYSQL_BASE_COLLECTIONS",
    "REQUIRED_CHROMA_COLLECTIONS",
    "chroma_readiness_report",
]


def chroma_readiness_report() -> dict[str, Any]:
    from src.algorithm.milvus_subprocess import vector_store_backend
    from src.knowledge.chroma_store import (
        chroma_client_mode,
        chroma_persist_directory,
        list_chroma_collections,
    )

    backend = vector_store_backend()
    if backend != "chroma":
        return {
            "backend": backend,
            "ready": True,
            "detail": f"当前 backend={backend}，不走 Chroma 集合检查",
            "missing": [],
            "present": [],
        }

    try:
        names = set(list_chroma_collections())
    except Exception as exc:
        return {
            "backend": "chroma",
            "mode": chroma_client_mode(),
            "persist_directory": str(chroma_persist_directory()),
            "ready": False,
            "detail": f"无法连接 Chroma：{exc}",
            "missing": list(REQUIRED_CHROMA_COLLECTIONS),
            "present": [],
            "bootstrap": "uv run python scripts/bootstrap_chroma_vectors.py",
        }

    missing = [name for name in REQUIRED_CHROMA_COLLECTIONS if name not in names]
    present = [name for name in REQUIRED_CHROMA_COLLECTIONS if name in names]
    mode = chroma_client_mode()
    loc = (
        f"Http {chroma_persist_directory()}（Docker 卷，见 deploy/chroma/docker-compose.yml）"
        if mode == "http"
        else f"嵌入式 @ {chroma_persist_directory()}"
    )
    detail = f"{loc}，已有 {len(names)} 个 collection"
    if missing:
        detail += f"；缺 {', '.join(missing)} → 运行 scripts/bootstrap_chroma_vectors.py"
    return {
        "backend": "chroma",
        "mode": mode,
        "persist_directory": str(chroma_persist_directory()),
        "ready": len(missing) == 0,
        "detail": detail,
        "missing": missing,
        "present": present,
        "all_collections": sorted(names),
        "bootstrap": "uv run python scripts/bootstrap_chroma_vectors.py",
    }
