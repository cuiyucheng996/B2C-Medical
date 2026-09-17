"""RAG 增强：yaml 默认 + 进程内覆盖 + 评测 ContextVar。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from algorithm.rag.types import RagPipelineConfig

__all__ = [
    "get_rag_config",
    "is_rag_enhanced",
    "rag_enhance_override",
    "reset_rag_enhance",
    "set_rag_config",
]

_OVERRIDE: dict[str, Any] = {}
_CTX: ContextVar[bool | None] = ContextVar("rag_enhance_ctx", default=None)


def rag_enhance_override(enabled: bool | None) -> Any:
    return _CTX.set(enabled)


def reset_rag_enhance(token: Any) -> None:
    _CTX.reset(token)


def is_rag_enhanced() -> bool:
    ctx = _CTX.get()
    if ctx is not None:
        return bool(ctx)
    return bool(get_rag_config().enhanced)


def _yaml_block() -> dict[str, Any]:
    try:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.configs import get_yaml_config

        block = (get_yaml_config() or {}).get("rag_enhance")
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def get_rag_config() -> RagPipelineConfig:
    y = _yaml_block()
    row = {**y, **_OVERRIDE}
    return RagPipelineConfig(
        enhanced=bool(row.get("enabled", row.get("enhanced", True))),
        multi_query=bool(row.get("multi_query", True)),
        max_queries=int(row.get("max_queries") or 3),
        rrf_k=int(row.get("rrf_k") or 60),
        fetch_k=int(row.get("fetch_k") or 12),
        top_k=int(row.get("top_k") or 8),
        use_cache=bool(row.get("use_cache", True)),
        cache_ttl_seconds=int(row.get("cache_ttl_seconds") or 300),
        keyword_rerank=bool(row.get("keyword_rerank", True)),
    )


def set_rag_config(patch: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(patch or {})
    if "enabled" in data:
        data["enhanced"] = data.pop("enabled")
    for key in (
        "enhanced",
        "multi_query",
        "max_queries",
        "rrf_k",
        "fetch_k",
        "top_k",
        "use_cache",
        "cache_ttl_seconds",
        "keyword_rerank",
    ):
        if key in data and data[key] is not None:
            _OVERRIDE[key] = data[key]
    cfg = get_rag_config()
    return {
        "enabled": cfg.enhanced,
        "multi_query": cfg.multi_query,
        "max_queries": cfg.max_queries,
        "rrf_k": cfg.rrf_k,
        "fetch_k": cfg.fetch_k,
        "top_k": cfg.top_k,
        "use_cache": cfg.use_cache,
        "cache_ttl_seconds": cfg.cache_ttl_seconds,
        "keyword_rerank": cfg.keyword_rerank,
    }
