"""公共知识评测：细意图 → 抽实体对齐 → 绑定工具查询。"""

from __future__ import annotations

import json
import time
from typing import Any

from algorithm.companion_turn import resolve_companion_turn
from algorithm.intent_catalog import INTENT_ROWS
from algorithm.intent_classify import CompanionIntentResult
from algorithm.kg_query import query_consult_graph, search_chroma_knowledge, search_exam_handbook
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
    "get_kb_eval_status",
]

_KB_TOOL_MAP = {str(tool.name): tool for tool in KNOWLEDGE_TOOLKIT.get_tools()}
_DEFAULT_HANDLER_BY_TOOL: dict[str, str] = {
    "query_consult_graph": "kg_part_diseases",
    "search_exam_handbook": "kg_exam_meaning",
    "search_chroma_knowledge": "kg_symptom_care",
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
    {
        "id": "complex_multi_part",
        "text": "请问帮我查一下，头部和胸部分别常见有哪些病，另外头痛一般算哪种病？",
    },
    {
        "id": "complex_symptom_exam_lab",
        "text": "我这两天嗓子疼还低烧，想问问血常规是查什么的，空腹血糖正常参考范围又是多少？",
    },
    {
        "id": "complex_hypertension_plain",
        "text": "麻烦用大白话讲讲高血压是什么病，它平时有哪些典型症状，生活上是不是一定要控盐？",
    },
    {
        "id": "complex_dual_lab",
        "text": "转氨酶和肌酐的正常值范围分别是多少？还有，转氨酶这项化验主要是查哪方面的？",
    },
    {
        "id": "complex_dizzy_two",
        "text": "头晕一般常见于哪些部位的问题？要是感冒引起的头晕，生活上要注意什么？",
    },
    {
        "id": "complex_polite_parallel",
        "text": "帮我查一下，头部常见有哪些病，顺便问脑卒中是不是就是中风的大白话说法？",
    },
    {
        "id": "complex_negation_exam",
        "text": "不是胃疼是胸口闷，心电图这种检查是查什么的，还有胸部常见有哪些病？",
    },
    {
        "id": "complex_syndrome",
        "text": "肝郁气滞证常见有哪些表现？和它有关的话，头部会不会也经常疼？",
    },
    {
        "id": "complex_long_stack",
        "text": "最近老觉得乏力、口干，想同时了解乏力常见和哪些病有关、口干生活上要注意什么、以及要不要先查个血常规是查什么的。",
    },
    {
        "id": "complex_coref_follow",
        "text": "我头一直疼，这个检查血常规能看出来吗？还有头疼本身常见出现在哪些部位？",
    },
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
            "pack": packed,
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
            "pack": packed,
        }
        hits = _from_items(list(packed.get("items") or []), channel="ragflow")
        if not hits:
            hits = retrieve_hits(query, top_k=top_k)
        return hits, "ragflow", extra
    packed = search_chroma_knowledge(query=query, handler=handler)
    extra = {
        "seeds": packed.get("seeds") or [],
        "summary": packed.get("summary") or "",
        "alignments": alignments,
        "pack": packed,
    }
    return _from_items(list(packed.get("items") or []), channel="chroma"), "chroma", extra


def _kb_item_from_turn(turn: dict[str, Any]) -> CompanionIntentResult | None:
    for item in list(turn.get("items") or []):
        if getattr(item, "major", "") == _MAJOR:
            return item
    items = list(turn.get("items") or [])
    return items[0] if items else None


def get_kb_eval_status() -> dict[str, Any]:
    from algorithm.companion_warmup import get_companion_nlu_status

    return get_companion_nlu_status()


def run_kb_eval(*, query: str, top_k: int = 8) -> dict[str, Any]:
    raw = (query or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty", "samples": KB_SAMPLES}
    t_turn = time.perf_counter()
    turn = resolve_companion_turn(raw, backend="llm", apply_file_route=False)
    turn_ms = int((time.perf_counter() - t_turn) * 1000)
    prep_row = turn.get("preprocess") if isinstance(turn.get("preprocess"), dict) else {}
    stage_ms = prep_row.get("stage_ms") if isinstance(prep_row.get("stage_ms"), dict) else {}
    timing: dict[str, Any] = {"extract_ms": turn_ms}
    for key in ("intent_ms", "catalog_ms", "extract_align_ms"):
        if stage_ms.get(key) is not None:
            timing[key] = int(stage_ms[key])
    if not turn.get("ok"):
        timing["total_ms"] = turn_ms
        return {"ok": False, "reason": turn.get("reason") or "turn_failed", "samples": KB_SAMPLES, "timing": timing}
    hit = _kb_item_from_turn(turn)
    if hit is None:
        return {"ok": False, "reason": "no_intent", "samples": KB_SAMPLES}
    query_text = str(turn.get("text") or raw)
    fine = str(hit.intent or "未细分")
    handler = str(hit.handler or "fallback")
    tools = [str(getattr(t, "name", "") or "") for t in tools_for_intent(major=hit.major, handler=handler)]
    prep_row = turn.get("preprocess") if isinstance(turn.get("preprocess"), dict) else {}
    mentions = prep_row.get("mentions") if isinstance(prep_row.get("mentions"), dict) else {}
    alignments = list(prep_row.get("alignments") or [])
    routed_major = str(hit.major or "")
    k = max(1, min(int(top_k or 8), 12))
    if routed_major != _MAJOR:
        timing["total_ms"] = turn_ms
        return {
            "ok": False,
            "reason": "not_kb_major",
            "major": routed_major,
            "text": raw,
            "query_after_preprocess": query_text,
            "intent": fine,
            "handler": handler,
            "tools": tools,
            "samples": KB_SAMPLES,
            "timing": timing,
        }
    t_recall = time.perf_counter()
    hits, channel, extra = _recall(handler, query_text, k, mentions, alignments)
    timing["recall_ms"] = int((time.perf_counter() - t_recall) * 1000)
    packed = extra.get("pack") if isinstance(extra.get("pack"), dict) else {}
    from algorithm.kb_eval_flow import build_kb_eval_flow

    flow = build_kb_eval_flow(
        raw=raw,
        query_text=query_text,
        intent=fine,
        handler=handler,
        intent_source=str(hit.source or ""),
        tools=tools,
        mentions=mentions,
        alignments=list(extra.get("alignments") or alignments),
        timing=timing,
        packed=packed,
        channel=channel,
    )
    t_rerank = time.perf_counter()
    reranked = _rerank(raw, hits)
    timing["rerank_ms"] = int((time.perf_counter() - t_rerank) * 1000)
    timing["total_ms"] = turn_ms + int(timing["recall_ms"]) + int(timing["rerank_ms"])
    chroma_bound = "search_chroma_knowledge" in tools
    chroma_queried = channel == "chroma"
    chroma_hits = len(hits) if chroma_queried else 0
    if chroma_queried:
        chroma_detail = f"已查询，返回 {chroma_hits} 条"
    elif chroma_bound:
        chroma_detail = f"工具已绑定，本轮走 {channel}，未查 Chroma"
    else:
        chroma_detail = f"本轮未绑定、未查询（通道 {channel}）"
    return {
        "ok": bool(hits),
        "major": _MAJOR,
        "text": raw,
        "query_after_preprocess": query_text,
        "intent": fine,
        "handler": handler,
        "intent_source": str(hit.source or ""),
        "tools": tools,
        "channel": channel,
        "mentions": mentions,
        "seeds": extra.get("seeds") or [],
        "summary": extra.get("summary") or "",
        "alignments": extra.get("alignments") or alignments,
        "chroma": {
            "queried": chroma_queried,
            "bound": chroma_bound,
            "hits": chroma_hits,
            "detail": chroma_detail,
        },
        "rerank_rule": "图谱按返回序；手册块按词命中重排",
        "retrieve_error": "",
        "kb": {"reachable": channel != "ragflow" or bool(hits), "error": ""},
        "recall": hits,
        "rerank": reranked,
        "samples": KB_SAMPLES,
        "timing": timing,
        "flow": flow,
        "rag": packed.get("rag") if isinstance(packed.get("rag"), dict) else None,
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


def run_kb_tool_probe(
    *,
    query: str,
    tools: list[str],
    handler: str = "",
    rag_enhanced: bool | None = None,
) -> dict[str, Any]:
    """陪聊进线解析后，按勾选工具直调（与 Agent 绑定工具同源）。"""
    raw = (query or "").strip()
    allowed = set(KNOWLEDGE_TOOLKIT.names())
    picked = [name for name in (tools or []) if name in allowed]
    if not raw:
        return {"ok": False, "reason": "empty", "available_tools": sorted(allowed)}
    if not picked:
        return {"ok": False, "reason": "no_tools", "available_tools": sorted(allowed)}
    base_handler = (handler or "").strip()
    t0 = time.perf_counter()
    turn = resolve_companion_turn(raw, backend="llm", apply_file_route=False)
    extract_ms = int((time.perf_counter() - t0) * 1000)
    if not turn.get("ok"):
        return {"ok": False, "reason": turn.get("reason") or "turn_failed", "available_tools": sorted(allowed)}
    kb_hit = _kb_item_from_turn(turn)
    prep_row = turn.get("preprocess") if isinstance(turn.get("preprocess"), dict) else {}
    query_text = str(turn.get("text") or raw).strip()
    mentions = prep_row.get("mentions") if isinstance(prep_row.get("mentions"), dict) else {}
    alignments = list(prep_row.get("alignments") or [])
    if not base_handler and kb_hit is not None:
        base_handler = str(kb_hit.handler or "")
    base_turn = {
        "query": query_text,
        "handler": base_handler or "kg_part_diseases",
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
    stage_ms = prep_row.get("stage_ms") if isinstance(prep_row.get("stage_ms"), dict) else {}
    timing: dict[str, Any] = {
        "extract_ms": extract_ms,
        "tools_ms": tools_ms,
        "total_ms": total_ms,
    }
    for key in ("intent_ms", "catalog_ms", "extract_align_ms"):
        if stage_ms.get(key) is not None:
            timing[key] = int(stage_ms[key])
    return {
        "ok": True,
        "major": _MAJOR,
        "text": query_text,
        "handler": base_handler or base_turn["handler"],
        "mentions": mentions,
        "alignments": alignments,
        "negated": list(prep_row.get("negated") or []),
        "entity_extracted": bool(prep_row.get("entity_extracted")),
        "routed_major": getattr(kb_hit, "major", "") if kb_hit else "",
        "routed_intent": getattr(kb_hit, "intent", "") if kb_hit else "",
        "tool_results": tool_rows,
        "rag_enhanced": effective_rag,
        "timing": timing,
        "available_tools": sorted(allowed),
    }
