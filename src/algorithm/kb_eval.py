"""公共知识评测：细意图 → 抽实体对齐 → 绑定工具查询。"""

from __future__ import annotations

import json
import time
from typing import Any

from algorithm.chat_preprocess import CompanionPreprocess, finish_preprocess_after_intent
from algorithm.intent_catalog import INTENT_BY_NAME, INTENT_ROWS
from algorithm.intent_classify import CompanionIntentResult, classify_fine_for_major
from algorithm.kg_query import query_consult_graph, search_exam_handbook, search_milvus_knowledge
from algorithm.pdf_kb import _query_terms
from algorithm.ragflow_kb import retrieve_hits
from tools.companion_tools import companion_turn, tools_for_intent
from tools.toolkits import KNOWLEDGE_TOOLKIT

__all__ = [
    "KB_SAMPLES",
    "list_kb_probe_handlers",
    "list_kb_probe_tools",
    "run_kb_eval",
    "run_kb_tool_probe",
]

_KB_TOOL_MAP = {str(tool.name): tool for tool in KNOWLEDGE_TOOLKIT.get_tools()}
_DEFAULT_HANDLER_BY_TOOL: dict[str, str] = {
    "query_consult_graph": "kg_part_diseases",
    "search_exam_handbook": "kg_exam_meaning",
    "search_milvus_knowledge": "kg_symptom_care",
}

_MAJOR = "公共知识问答"
_GRAPH = frozenset(
    {
        "kg_part_diseases",
        "kg_disease_symptoms",
        "kg_symptom_parts",
        "kg_disease_plain",
        "kg_syndrome_signs",
    }
)
_HANDBOOK = frozenset({"kg_exam_meaning", "kg_lab_range", "kg_symptom_care"})

KB_SAMPLES: list[dict[str, str]] = [
    {"id": "part", "text": "头部常见有哪些病"},
    {"id": "plain", "text": "头痛是什么疾病"},
    {"id": "exam_cbc", "text": "血常规这个检查是查什么的"},
    {"id": "exam_alt", "text": "转氨酶正常值是多少"},
    {"id": "exam_glu", "text": "空腹血糖参考范围是多少"},
]


def _rerank(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens = [tok for tok in _query_terms(query) if len(tok) >= 3]
    ranked: list[dict[str, Any]] = []
    for row in hits:
        text = str(row.get("text") or "")
        kw = sum(text.count(tok) for tok in tokens)
        recall = float(row.get("score") or 0)
        item = dict(row)
        item["keyword_hits"] = kw
        item["rerank_score"] = round(recall + kw * 0.5, 4)
        ranked.append(item)
    ranked.sort(key=lambda item: (-float(item.get("rerank_score") or 0), int(item.get("recall_rank") or 0)))
    for idx, item in enumerate(ranked, start=1):
        item["rerank_rank"] = idx
        item["moved"] = int(item.get("recall_rank") or 0) - idx
    return ranked


def _from_items(items: list[str], *, channel: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for idx, text in enumerate(items, start=1):
        line = str(text or "").strip()
        if not line:
            continue
        hits.append(
            {
                "text": line[:400],
                "doc": "",
                "score": 1.0,
                "recall_rank": idx,
                "channel": channel,
            }
        )
    return hits


def _recall(
    handler: str,
    query: str,
    top_k: int,
    mentions: dict[str, list[str]],
    alignments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    extra: dict[str, Any] = {}
    if handler in _GRAPH:
        packed = query_consult_graph(
            handler=handler,
            query=query,
            mentions=mentions,
            alignments=alignments,
        )
        extra = {
            "seeds": packed.get("seeds") or [],
            "summary": packed.get("summary") or "",
            "alignments": packed.get("alignments") or alignments,
        }
        return _from_items(list(packed.get("items") or []), channel="neo4j"), "neo4j", extra
    if handler in _HANDBOOK:
        packed = search_exam_handbook(
            query=query,
            mentions=mentions,
            alignments=alignments,
            handler=handler,
        )
        extra = {
            "seeds": packed.get("seeds") or [],
            "summary": packed.get("summary") or "",
            "alignments": packed.get("alignments") or alignments,
        }
        hits = _from_items(list(packed.get("items") or []), channel="ragflow")
        if not hits:
            hits = retrieve_hits(query, top_k=top_k)
        return hits, "ragflow", extra
    packed = search_milvus_knowledge(query=query, handler=handler)
    extra = {
        "seeds": packed.get("seeds") or [],
        "summary": packed.get("summary") or "",
        "alignments": alignments,
    }
    return _from_items(list(packed.get("items") or []), channel="milvus"), "milvus", extra


def run_kb_eval(*, query: str, top_k: int = 8) -> dict[str, Any]:
    raw = (query or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty", "samples": KB_SAMPLES}
    hit = classify_fine_for_major(raw, major=_MAJOR, llm_seconds=5)
    fine = str(hit.intent or "未细分")
    handler = str(hit.handler or (INTENT_BY_NAME.get(fine) or {}).get("handler") or "fallback")
    tools = [str(getattr(t, "name", "") or "") for t in tools_for_intent(major=_MAJOR, handler=handler)]
    prep = finish_preprocess_after_intent(
        CompanionPreprocess(raw_text=raw, corrected_text=raw, resolved_text=raw),
        hit,
    )
    mentions = prep.mentions if isinstance(prep.mentions, dict) else {}
    alignments = list(prep.alignments or [])
    k = max(1, min(int(top_k or 8), 12))
    hits, channel, extra = _recall(handler, raw, k, mentions, alignments)
    reranked = _rerank(raw, hits)
    milvus_bound = "search_milvus_knowledge" in tools
    milvus_queried = channel == "milvus"
    milvus_hits = len(hits) if milvus_queried else 0
    if milvus_queried:
        milvus_detail = f"已查询，返回 {milvus_hits} 条"
    elif milvus_bound:
        milvus_detail = f"工具已绑定，本轮走 {channel}，未查 Milvus"
    else:
        milvus_detail = f"本轮未绑定、未查询（通道 {channel}）"
    return {
        "ok": bool(hits),
        "major": _MAJOR,
        "text": raw,
        "intent": fine,
        "handler": handler,
        "intent_source": str(hit.source or ""),
        "tools": tools,
        "channel": channel,
        "mentions": mentions,
        "seeds": extra.get("seeds") or [],
        "summary": extra.get("summary") or "",
        "alignments": extra.get("alignments") or alignments,
        "milvus": {
            "queried": milvus_queried,
            "bound": milvus_bound,
            "hits": milvus_hits,
            "detail": milvus_detail,
        },
        "rerank_rule": "图谱按返回序；手册块按词命中重排",
        "retrieve_error": "",
        "kb": {"reachable": channel != "ragflow" or bool(hits), "error": ""},
        "recall": hits,
        "rerank": reranked,
        "samples": KB_SAMPLES,
    }


def list_kb_probe_tools() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tool, meta in KNOWLEDGE_TOOLKIT.items:
        rows.append(
            {
                "name": str(tool.name),
                "title": str(meta.title or tool.name),
                "summary": str(meta.summary or ""),
            }
        )
    return rows


def list_kb_probe_handlers() -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for item in INTENT_ROWS:
        if str(item.get("major") or "") != _MAJOR:
            continue
        handler = str(item.get("handler") or "").strip()
        if not handler or handler in seen:
            continue
        seen.add(handler)
        rows.append({"handler": handler, "intent": str(item.get("intent") or "")})
    return rows


def _extract_kb_entities(query: str, *, handler: str) -> CompanionPreprocess:
    raw = (query or "").strip()
    hit = CompanionIntentResult(
        intent="工具直测",
        major=_MAJOR,
        confidence=1.0,
        source="kb_tool_probe",
        reason="跳过细意图 LLM，只抽实体并对齐",
        slots=(),
        handler=(handler or "kg_part_diseases").strip() or "kg_part_diseases",
        recommend_action="正常处理",
    )
    return finish_preprocess_after_intent(
        CompanionPreprocess(raw_text=raw, corrected_text=raw, resolved_text=raw),
        hit,
    )


def run_kb_tool_probe(
    *,
    query: str,
    tools: list[str],
    handler: str = "",
    rag_enhanced: bool | None = None,
) -> dict[str, Any]:
    """先抽实体对齐，再按用户勾选的工具直调（不走细意图 LLM、不走 Agent）。"""
    raw = (query or "").strip()
    allowed = set(KNOWLEDGE_TOOLKIT.names())
    picked = [name for name in (tools or []) if name in allowed]
    if not raw:
        return {"ok": False, "reason": "empty", "available_tools": sorted(allowed)}
    if not picked:
        return {"ok": False, "reason": "no_tools", "available_tools": sorted(allowed)}
    base_handler = (handler or "").strip()
    t0 = time.perf_counter()
    prep = _extract_kb_entities(raw, handler=base_handler or "kg_part_diseases")
    extract_ms = int((time.perf_counter() - t0) * 1000)
    query_text = (prep.resolved_text or prep.corrected_text or raw).strip()
    mentions = prep.mentions if isinstance(prep.mentions, dict) else {}
    alignments = list(prep.alignments or [])
    base_turn = {
        "query": query_text,
        "handler": base_handler or str(prep.intent_handler or "kg_part_diseases"),
        "mentions": mentions,
        "alignments": alignments,
        "user_id": "eval",
    }
    from algorithm.rag.config import get_rag_config, rag_enhance_override, reset_rag_enhance

    rag_token = rag_enhance_override(rag_enhanced)
    tool_rows: list[dict[str, Any]] = []
    tools_ms = 0
    try:
        for name in picked:
            tool = _KB_TOOL_MAP.get(name)
            if tool is None:
                tool_rows.append({"tool": name, "ok": False, "error": "unknown_tool", "elapsed_ms": 0})
                continue
            tool_handler = base_handler or _DEFAULT_HANDLER_BY_TOOL.get(name) or base_turn["handler"]
            tok = companion_turn.set({**base_turn, "handler": tool_handler})
            t_tool = time.perf_counter()
            try:
                out = tool.invoke({"query": query_text})
            except Exception as exc:
                elapsed = int((time.perf_counter() - t_tool) * 1000)
                tools_ms += elapsed
                tool_rows.append(
                    {
                        "tool": name,
                        "handler": tool_handler,
                        "ok": False,
                        "error": str(exc),
                        "elapsed_ms": elapsed,
                    }
                )
                companion_turn.reset(tok)
                continue
            elapsed = int((time.perf_counter() - t_tool) * 1000)
            tools_ms += elapsed
            companion_turn.reset(tok)
            parsed: dict[str, Any]
            if isinstance(out, str):
                try:
                    parsed = json.loads(out)
                except json.JSONDecodeError:
                    parsed = {"ok": True, "summary": out, "data": {}}
            elif isinstance(out, dict):
                parsed = out
            else:
                parsed = {"ok": True, "summary": str(out), "data": {}}
            tool_rows.append(
                {
                    "tool": name,
                    "handler": tool_handler,
                    "ok": bool(parsed.get("ok", True)),
                    "usable": parsed.get("usable"),
                    "summary": parsed.get("summary") or "",
                    "data": parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed,
                    "elapsed_ms": elapsed,
                }
            )
    finally:
        reset_rag_enhance(rag_token)
    total_ms = extract_ms + tools_ms
    cfg = get_rag_config()
    effective_rag = rag_enhanced if rag_enhanced is not None else cfg.enhanced
    return {
        "ok": True,
        "major": _MAJOR,
        "text": query_text,
        "handler": base_handler or base_turn["handler"],
        "mentions": mentions,
        "alignments": alignments,
        "negated": list(prep.negated or []),
        "entity_extracted": bool(prep.entity_extracted),
        "tool_results": tool_rows,
        "rag_enhanced": effective_rag,
        "timing": {
            "extract_ms": extract_ms,
            "tools_ms": tools_ms,
            "total_ms": total_ms,
        },
        "available_tools": sorted(allowed),
    }
