"""Query → 多路召回 → RRF → 重排 → 缓存。"""

from __future__ import annotations

import time
from typing import Any

from algorithm.rag.cache import get_cached_hits, set_cached_hits
from algorithm.rag.config import get_rag_config
from algorithm.rag.fuse import rrf_merge
from algorithm.rag.query_stage import expand_queries
from algorithm.rag.rank import keyword_rerank_hits
from algorithm.rag.retrievers import hits_from_text_items, retrieve_chroma_hits, retrieve_handbook_hits
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
    raw_count = len(items)
    hits = hits_from_text_items(items, channel="graph")
    rerank_applied = bool(cfg.keyword_rerank and hits)
    if rerank_applied:
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
            "main_query": (query or "").strip(),
            "note": "图谱单路召回 + 关键词重排",
            "timing_ms": {"pipeline_ms": timing},
            "hits": [h.to_dict() for h in hits],
            "cache_detail": "图谱增强不使用 RAG 结果缓存",
            "query_expand_detail": "不做多 query（enhance_graph_pack 仅重排 Neo4j items）",
            "recall_detail": f"Neo4j 已有 {raw_count} 条 → 转为 RagHit 列表",
            "rrf_detail": "不适用（单路，无 RRF）",
            "rerank_detail": (
                f"keyword_rerank={'开' if rerank_applied else '关'}；"
                f"{raw_count} 条 → top_k={cfg.top_k} 共 {len(hits)} 条"
            ),
            "output_detail": f"写回 items/summary；summary 长度 {len(summary)}",
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
    source = {"handbook": "ragflow", "chroma": "chroma", "graph": "neo4j"}[channel]

    if channel == "handbook":
        cache_q = main_q
    else:
        cache_q = (query or "").strip()

    cache_detail = (
        f"未启用（use_cache={cfg.use_cache}）"
        if not cfg.use_cache
        else f"键 channel={channel} handler={name} query={cache_q[:80]}… TTL={cfg.cache_ttl_seconds}s"
    )

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
                    "main_query": main_q,
                    "cache_key": cache_q,
                    "note": "cache_hit",
                    "timing_ms": {"pipeline_ms": 0, "cache": True},
                    "hits": cached[: cfg.top_k],
                    "cache_detail": cache_detail + " → 命中，跳过后续召回",
                    "query_expand_detail": "缓存命中，未扩展",
                    "recall_detail": "跳过",
                    "rrf_detail": "跳过",
                    "rerank_detail": "跳过",
                    "output_detail": f"直接返回 {len(items)} 条 items",
                },
            )
        cache_detail = cache_detail + " → 未命中"

    expand_base = main_q if channel == "handbook" else cache_q
    queries = expand_queries(expand_base, seeds, config=cfg)
    from algorithm.kb_eval_display import numbered_lines

    query_expand_detail = (
        f"multi_query={cfg.multi_query} max={cfg.max_queries} → {len(queries)} 条待检索：\n"
        + numbered_lines(queries)
    )
    lists: list[list[RagHit]] = []
    recall_lines: list[str] = []
    for idx, q in enumerate(queries, start=1):
        snippet = q if len(q) <= 48 else q[:47] + "…"
        if channel == "handbook":
            lst = retrieve_handbook_hits(q, top_k=cfg.fetch_k)
            lists.append(lst)
            recall_lines.append(f"{idx}. RAGFlow「{snippet}」→ {len(lst)} 条")
        elif channel == "chroma":
            lst = retrieve_chroma_hits(q)
            lists.append(lst)
            recall_lines.append(f"{idx}. Chroma「{snippet}」→ {len(lst)} 条")
        else:
            break
    recall_detail = "；".join(recall_lines) if recall_lines else "无"

    if channel in {"handbook", "chroma"}:
        rrf_applied = len(lists) > 1
        merged = rrf_merge(lists, k=cfg.rrf_k) if rrf_applied else (lists[0] if lists else [])
        rrf_detail = (
            f"RRF 融合 k={cfg.rrf_k}，{len(lists)} 路 → {len(merged)} 条"
            if rrf_applied
            else f"单路 {len(merged)} 条，未做 RRF"
        )
        rerank_applied = bool(cfg.keyword_rerank and merged)
        if rerank_applied:
            merged = keyword_rerank_hits(main_q if channel == "handbook" else cache_q, merged)
        hits = merged[: cfg.top_k]
        hit_dicts = [h.to_dict() for h in hits]
        if cfg.use_cache and hit_dicts:
            set_cached_hits(channel, cache_q, name, hit_dicts, ttl=cfg.cache_ttl_seconds)
        items = [h.text for h in hits]
        note = "多查询+RRF+重排" if len(queries) > 1 else "单查询+重排"
        rerank_detail = (
            f"keyword_rerank={'开' if rerank_applied else '关'}；取 top_k={cfg.top_k} 得 {len(hits)} 条"
        )
        output_detail = f"写回 items（{len(items)}）+ summary + pack.rag"
    else:
        hits = []
        hit_dicts = []
        items = []
        note = "unsupported_channel"
        rrf_detail = "-"
        rerank_detail = "-"
        output_detail = "-"

    timing = int((time.perf_counter() - t0) * 1000)
    pack = _pack(name, seeds, items, _summarize(name, seeds, items), source, alignments=alignments or [])
    return _attach_rag(
        pack,
        meta={
            "enhanced": True,
            "channel": channel,
            "queries": queries,
            "main_query": main_q,
            "cache_key": cache_q,
            "note": note,
            "timing_ms": {"pipeline_ms": timing, "cache": False},
            "hits": hit_dicts,
            "cache_detail": cache_detail,
            "query_expand_detail": query_expand_detail,
            "recall_detail": recall_detail,
            "rrf_detail": rrf_detail,
            "rerank_detail": rerank_detail,
            "output_detail": output_detail,
        },
    )
