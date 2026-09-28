"""百炼知识库 Retrieve（运营/合规文档，与 Neo4j/Chroma 图谱互补）。"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["bailian_kb_config", "bailian_retrieve", "bailian_list_indices"]


def _yaml_aliyun() -> dict[str, Any]:
    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp")
    if not isinstance(block, dict):
        return {}
    aliyun = block.get("aliyun")
    return dict(aliyun) if isinstance(aliyun, dict) else {}


def bailian_kb_config() -> dict[str, Any]:
    cfg = _yaml_aliyun()
    workspace = (
        os.getenv("BAILIAN_WORKSPACE_ID")
        or str(cfg.get("bailian_workspace_id") or "").strip()
    )
    index_id = os.getenv("BAILIAN_INDEX_ID") or str(cfg.get("bailian_index_id") or "").strip()
    endpoint = str(cfg.get("bailian_endpoint") or "bailian.cn-beijing.aliyuncs.com").strip()
    ak = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID") or os.getenv("ALIYUN_ACCESS_KEY_ID") or ""
    sk = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_ACCESS_KEY_SECRET") or ""
    return {
        "workspace_id": workspace,
        "index_id": index_id,
        "endpoint": endpoint,
        "has_ak": bool(ak and sk),
    }


def _client():
    from alibabacloud_bailian20231229.client import Client as BailianClient
    from alibabacloud_tea_openapi import models as open_api_models

    cfg = bailian_kb_config()
    ak = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID") or os.getenv("ALIYUN_ACCESS_KEY_ID") or ""
    sk = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET") or os.getenv("ALIYUN_ACCESS_KEY_SECRET") or ""
    if not ak or not sk:
        raise RuntimeError(
            "百炼 Retrieve 需要 ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET（或 ALIYUN_ACCESS_KEY_*）"
        )
    config = open_api_models.Config(access_key_id=ak, access_key_secret=sk)
    config.endpoint = cfg["endpoint"]
    return BailianClient(config), cfg["workspace_id"]


def bailian_retrieve(
    query: str,
    *,
    index_id: str | None = None,
    dense_top_k: int = 5,
    enable_rerank: bool = True,
) -> dict[str, Any]:
    text = (query or "").strip()
    if not text:
        return {"success": False, "error": "query 为空", "nodes": []}
    cfg = bailian_kb_config()
    workspace = cfg["workspace_id"]
    idx = (index_id or cfg["index_id"] or "").strip()
    if not workspace or not idx:
        return {
            "success": False,
            "error": "请配置 BAILIAN_WORKSPACE_ID 与 BAILIAN_INDEX_ID（或 config.yaml mcp.aliyun）",
            "nodes": [],
        }
    try:
        from alibabacloud_bailian20231229 import models as bailian_models

        client, ws = _client()
        req = bailian_models.RetrieveRequest(
            index_id=idx,
            query=text,
            dense_similarity_top_k=max(1, min(int(dense_top_k), 20)),
            enable_reranking=bool(enable_rerank),
        )
        resp = client.retrieve(ws, req)
        body = resp.body.to_map() if resp.body else {}
        nodes = body.get("Data") or body.get("data") or body
        return {"success": True, "index_id": idx, "query": text, "raw": body, "summary": _summarize(nodes)}
    except Exception as exc:
        logger.warning("bailian retrieve failed", exc_info=True)
        return {"success": False, "error": str(exc)[:500], "nodes": []}


def bailian_list_indices(*, page: int = 1, size: int = 20) -> dict[str, Any]:
    cfg = bailian_kb_config()
    workspace = cfg["workspace_id"]
    if not workspace:
        return {"success": False, "error": "缺少 BAILIAN_WORKSPACE_ID", "indices": []}
    try:
        from alibabacloud_bailian20231229 import models as bailian_models

        client, ws = _client()
        req = bailian_models.ListIndicesRequest(
            page_number=str(max(1, page)),
            page_size=str(max(1, min(size, 50))),
        )
        resp = client.list_indices(ws, req)
        body = resp.body.to_map() if resp.body else {}
        return {"success": True, "raw": body}
    except Exception as exc:
        logger.warning("bailian list_indices failed", exc_info=True)
        return {"success": False, "error": str(exc)[:500], "indices": []}


def _summarize(payload: Any) -> str:
    try:
        if isinstance(payload, dict):
            nodes = payload.get("Nodes") or payload.get("nodes") or []
        elif isinstance(payload, list):
            nodes = payload
        else:
            return str(payload)[:2000]
        lines: list[str] = []
        for node in nodes[:8]:
            if not isinstance(node, dict):
                continue
            text = str(node.get("Text") or node.get("text") or node.get("content") or "")[:400]
            score = node.get("Score") or node.get("score")
            if text:
                lines.append(f"- ({score}) {text}")
        return "\n".join(lines) if lines else json.dumps(payload, ensure_ascii=False)[:2000]
    except Exception:
        return ""
