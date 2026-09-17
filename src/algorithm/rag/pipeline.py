"""Query → 多路召回 → RRF → 重排 → 缓存。"""

from __future__ import annotations

import time
from typing import Any

from algorithm.rag.cache import get_cached_hits, set_cached_hits
from algorithm.rag.config import get_rag_config
from algorithm.rag.fuse import rrf_merge
from algorithm.rag.query_stage import expand_queries
from algorithm.rag.rank import keyword_rerank_hits
from algorithm.rag.retrievers import hits_from_text_items, retrieve_handbook_hits, retrieve_milvus_hits
from algorithm.rag.types import RagChannel, RagHit

__all__ = ["run_rag_pipeline", "enhance_graph_pack"]


def _attach_rag(pack: dict[str, Any], *, meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(pack)
    out["rag"] = meta
    return out


def enhance_graph_pack(
    pack: dict[str, Any],
    *,
    query: str,
    handler: str,
) -> dict[str, Any]:
    cfg = get_rag_config()
    if not cfg.enhanced:
        return pack
    t0 = time.perf_counter()
    items = list(pack.get("items") or [])
    hits = hits_from_text_items(items, channel="graph")
    if cfg.keyword_rerank and hits:
        hits = keyword_rerank_hits(query, hits)
    hits = hits[: cfg.top_k]
    new_items = [h.text for h in hits]
    from algorithm.kg_query import _summarize

    seeds = list(pack.get("seeds") or [])
    summary = _summarize(handler, seeds, new_items)
    timing = int((time.perf_counter() - t0) * 1000)
    return _attach_rag(
        {
            **pack,
            "items": new_items[:24],
            "summary": summary,
        },
        meta={
            "enhanced": True,
            "channel": "graph",
            "queries": [query.strip()],
            "note": "图谱单路召回 + 关键词重排",
            "timing_ms": {"pipeline_ms": timing},
            "hits": [h.to_dict() for h in hits],
        },
    )


def run_rag_pipeline(
    channel: RagChannel,
    *,
    query: str,
    handler: str = "",
    mentions: dict[str, list[str]] | None = None,
    alignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from algorithm.kg_query import _pack, _resolve_seeds, _summarize

    cfg = get_rag_config()
    name = (handler or "").strip() or ("kg_exam_meaning" if channel == "handbook" else "fallback")
    seeds = _resolve_seeds(name, query, mentions or {}, alignments or [])
    needles = list(dict.fromkeys([*seeds, (query or "").strip()]))
    main_q = " ".join(item for item in needles if item)[:200]
    source = {"handbook": "ragflow", "milvus": "milvus", "graph": "neo4j"}[channel]

    if channel == "handbook":
        cache_q = main_q
    else:
        cache_q = (query or "").strip()

    t0 = time.perf_counter()
    if cfg.use_cache:
        cached = get_cached_hits(channel, cache_q, name, ttl=cfg.cache_ttl_seconds)
        if cached is not None:
            items = [str(row.get("text") or "") for row in cached if row.get("text")]
            pack = _pack(name, seeds, items, _summarize(name, seeds, items), source, alignments=alignments or [])
            return _attach_rag(
                pack,
                meta={
                    "enhanced": True,
                    "channel": channel,
                    "queries": [cache_q],
                    "note": "cache_hit",
                    "timing_ms": {"pipeline_ms": 0, "cache": True},
                    "hits": cached[: cfg.top_k],
                },
            )

    queries = expand_queries(main_q if channel == "handbook" else cache_q, seeds, config=cfg)
    lists: list[list[RagHit]] = []
    for q in queries:
        if channel == "handbook":
            lists.append(retrieve_handbook_hits(q, top_k=cfg.fetch_k))
        elif channel == "milvus":
            lists.append(retrieve_milvus_hits(q))
        else:
            break

    if channel in {"handbook", "milvus"}:
        merged = rrf_merge(lists, k=cfg.rrf_k) if len(lists) > 1 else (lists[0] if lists else [])
        if cfg.keyword_rerank and merged:
            merged = keyword_rerank_hits(main_q if channel == "handbook" else cache_q, merged)
        hits = merged[: cfg.top_k]
        hit_dicts = [h.to_dict() for h in hits]
        if cfg.use_cache and hit_dicts:
            set_cached_hits(channel, cache_q, name, hit_dicts, ttl=cfg.cache_ttl_seconds)
        items = [h.text for h in hits]
        note = "多查询+RRF+重排" if len(queries) > 1 else "单查询+重排"
    else:
        hits = []
        hit_dicts = []
        items = []
        note = "unsupported_channel"

    timing = int((time.perf_counter() - t0) * 1000)
    pack = _pack(name, seeds, items, _summarize(name, seeds, items), source, alignments=alignments or [])
    return _attach_rag(
        pack,
        meta={
            "enhanced": True,
            "channel": channel,
            "queries": queries,
            "note": note,
            "timing_ms": {"pipeline_ms": timing, "cache": False},
            "hits": hit_dicts,
        },
    )
