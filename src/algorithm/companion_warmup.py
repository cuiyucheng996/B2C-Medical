"""陪聊 NLU 预热：与线上请求同路径，不改变推理结果。"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root
from b2c_b2b_bridge import b2b_api_config, b2b_uie_extract, warmup_b2b_bridge

logger = logging.getLogger(__name__)

__all__ = [
    "get_companion_nlu_status",
    "warm_companion_nlu",
]

_startup_warm: dict[str, Any] | None = None


def warm_companion_nlu() -> dict[str, Any]:
    """加载对齐目录缓存 + UIE 单例，避免首条用户请求冷启动。"""
    global _startup_warm
    stats: dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat()}
    from algorithm.companion_cypher import warm_align_catalogs

    stats["catalog_ms"] = warm_align_catalogs()
    t0 = time.perf_counter()
    try:
        warmup_b2b_bridge()
        uie = b2b_uie_extract()
        uie.get_uie_predictor()
        uie.extract_entity("头部", ["部位", "疾病", "症状"])
        stats["uie_ok"] = True
    except Exception as exc:
        stats["uie_ok"] = False
        stats["uie_error"] = str(exc)
        logger.warning("companion UIE warm failed: %s", exc)
    stats["uie_ms"] = int((time.perf_counter() - t0) * 1000)
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    _startup_warm = dict(stats)
    logger.info("companion NLU warm done: %s", stats)
    return stats


def _uie_status() -> dict[str, Any]:
    row: dict[str, Any] = {
        "need_warm": True,
        "ready": False,
        "detail": "未加载",
        "device": "",
    }
    try:
        uie = b2b_uie_extract()
        info = uie.get_uie_predictor.cache_info()
        if info.currsize < 1:
            row["detail"] = "Predictor 未在进程内初始化"
            return row
        predictor = uie.get_uie_predictor()
        if predictor is None:
            row["detail"] = "uie_enabled=false 或权重缺失"
            return row
        row["ready"] = True
        row["detail"] = "单例已加载，与线上 infer 同路径"
        row["device"] = str(getattr(predictor, "device", "") or "")
    except Exception as exc:
        row["detail"] = str(exc)[:240]
    return row


def _chroma_status() -> dict[str, Any]:
    row: dict[str, Any] = {"need_warm": True, "ready": False, "detail": "未探测"}

    def _probe() -> dict[str, Any]:
        ensure_uvgraph_root()
        from src.knowledge.vector_store_readiness import chroma_readiness_report

        report = chroma_readiness_report()
        return {
            "need_warm": True,
            "ready": bool(report.get("ready")),
            "detail": str(report.get("detail") or ""),
            "collections": report.get("all_collections") or report.get("present") or [],
            "missing": report.get("missing") or [],
            "bootstrap": report.get("bootstrap") or "",
        }

    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_probe)
            out = future.result(timeout=4.0)
            row.update(out)
    except concurrent.futures.TimeoutError:
        row["detail"] = "连接探测超时（>4s），请检查 Chroma 服务或 persist_directory"
    except Exception as exc:
        row["detail"] = str(exc)[:240]
    return row


def _neo4j_status() -> dict[str, Any]:
    row: dict[str, Any] = {"need_warm": False, "ready": False, "detail": "未探测"}
    try:
        ensure_uvgraph_root()
        from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver

        driver = get_neo4j_driver()
        verify = getattr(driver, "verify_connectivity", None)
        if callable(verify):
            verify()
        row["ready"] = True
        row["detail"] = "Driver 可用（对齐目录与图谱查询）"
    except Exception as exc:
        row["detail"] = str(exc)[:240]
    return row


def _align_rerank_status() -> dict[str, Any]:
    row: dict[str, Any] = {"need_warm": False, "ready": True, "detail": ""}
    try:
        cfg = b2b_api_config().get_b2b_api_config()
        enabled = bool(cfg.get("align_rerank_enabled", True))
        k = int(cfg.get("align_rerank_recall_k") or 10)
        row["ready"] = True
        row["detail"] = (
            f"align_rerank.enabled={enabled}，recall_k={k}（按需 HTTP，无进程内模型预热）"
        )
    except Exception as exc:
        row["ready"] = False
        row["detail"] = str(exc)[:240]
    return row


def get_companion_nlu_status() -> dict[str, Any]:
    """评测台：各组件是否已预热 / 依赖是否就绪。"""
    from algorithm.companion_cypher import align_catalog_cache_status
    from algorithm.kb_eval_display import ms_to_sec_text

    cat = align_catalog_cache_status()
    uie = _uie_status()
    chroma = _chroma_status()
    neo4j = _neo4j_status()
    rerank = _align_rerank_status()
    counts = cat.get("counts") or {}
    cat_detail = (
        f"缓存 age={cat.get('age_sec')}s / TTL {cat.get('ttl_sec')}s · "
        f"部位{counts.get('body_part', 0)} 疾病{counts.get('disease', 0)} 症状{counts.get('symptom', 0)}"
        if cat.get("cached")
        else "未缓存，首条对齐会拉 Neo4j 三表"
    )
    components = [
        {
            "id": "startup_warm",
            "label": "服务启动预热",
            "need_warm": True,
            "ready": bool(_startup_warm and _startup_warm.get("uie_ok")),
            "detail": (
                f"catalog {ms_to_sec_text(_startup_warm.get('catalog_ms'))} · UIE {ms_to_sec_text(_startup_warm.get('uie_ms'))}"
                if _startup_warm
                else "本进程尚未执行 lifespan 预热（可点「立即预热」）"
            ),
            "meta": _startup_warm,
        },
        {
            "id": "align_catalog",
            "label": "对齐目录（Neo4j 三表）",
            "need_warm": True,
            "ready": bool(cat.get("ready")),
            "detail": cat_detail,
            "meta": cat,
        },
        {
            "id": "uie",
            "label": "UIE 实体抽离",
            "need_warm": True,
            "ready": bool(uie.get("ready")),
            "detail": str(uie.get("detail") or ""),
            "meta": uie,
        },
        {
            "id": "chroma_align",
            "label": "向量对齐（Chroma）",
            "need_warm": True,
            "ready": bool(chroma.get("ready")),
            "detail": str(chroma.get("detail") or ""),
            "meta": chroma,
        },
        {
            "id": "align_rerank",
            "label": "DeepSeek 对齐精排",
            "need_warm": False,
            "ready": bool(rerank.get("ready")),
            "detail": str(rerank.get("detail") or ""),
            "meta": rerank,
        },
        {
            "id": "neo4j",
            "label": "Neo4j 图谱 / Finding",
            "need_warm": False,
            "ready": bool(neo4j.get("ready")),
            "detail": str(neo4j.get("detail") or ""),
            "meta": neo4j,
        },
    ]
    return {
        "ok": True,
        "components": components,
        "pipeline_stages": [
            {"id": "intent_ms", "label": "意图 LLM"},
            {"id": "catalog_ms", "label": "对齐目录"},
            {"id": "extract_align_ms", "label": "UIE+否定+对齐"},
            {"id": "extract_ms", "label": "进线合计（含纠错+意图+抽对齐）"},
            {"id": "recall_ms", "label": "知识召回"},
            {"id": "rerank_ms", "label": "结果重排"},
            {"id": "tools_ms", "label": "工具 invoke"},
            {"id": "total_ms", "label": "服务端合计"},
        ],
        "startup_warm": _startup_warm,
    }
