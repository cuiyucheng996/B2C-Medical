"""公共知识评测：按序号展示 意图 → 抽离对齐 → RAG 增强 流程。"""

from __future__ import annotations

from typing import Any

from algorithm.kb_eval_display import format_query_plan, ms_to_sec_text, numbered_lines
from algorithm.rag.config import get_rag_config, is_rag_enhanced

__all__ = ["build_kb_eval_flow", "format_companion_rag_block"]


def _step(no: str, title: str, detail: str) -> dict[str, str]:
    return {"no": no, "title": title, "detail": detail}


def _mentions_line(mentions: dict[str, list[str]]) -> str:
    bits = []
    for key in ("部位", "疾病", "症状"):
        vals = [str(v).strip() for v in (mentions.get(key) or []) if str(v).strip()]
        if vals:
            bits.append(f"{key}：{'、'.join(vals)}")
    return "；".join(bits) if bits else "（无 span）"


def _align_line(alignments: list[dict[str, Any]]) -> str:
    rows = []
    for item in alignments or []:
        if not isinstance(item, dict):
            continue
        m = str(item.get("mention") or "").strip()
        s = str(item.get("standard_name") or "").strip()
        method = str(item.get("method") or "").strip()
        if m or s:
            rows.append(f"{m}→{s or '?'}" + (f"（{method}）" if method else ""))
    return "；".join(rows[:8]) if rows else "（无标准名锁定）"


def build_kb_eval_flow(
    *,
    raw: str,
    query_text: str,
    intent: str,
    handler: str,
    intent_source: str,
    tools: list[str],
    mentions: dict[str, list[str]],
    alignments: list[dict[str, Any]],
    timing: dict[str, Any],
    packed: dict[str, Any],
    channel: str,
) -> list[dict[str, Any]]:
    """返回 [{title, steps:[{no,title,detail}]}] 三块 + 说明。"""
    prep_timing = timing or {}
    rag = packed.get("rag") if isinstance(packed.get("rag"), dict) else {}
    cfg = get_rag_config()
    rag_on = is_rag_enhanced()
    seeds = list(packed.get("seeds") or [])
    source = str(packed.get("source") or channel or "")

    intent_steps = [
        _step(
            "1.1",
            "大类与细意图",
            f"公共知识问答 / {intent or '未细分'} / handler={handler or '-'}（来源 {intent_source or '-'}）",
        ),
        _step(
            "1.2",
            "绑定工具",
            "、".join(tools) if tools else "（无）",
        ),
        _step(
            "1.3",
            "意图耗时",
            f"意图 {ms_to_sec_text(prep_timing.get('intent_ms'))}，进线合计 {ms_to_sec_text(prep_timing.get('extract_ms'))}",
        ),
    ]

    extract_steps = [
        _step("2.1", "用户句 → 纠错/指代后", f"原句：{raw[:120]} → 检索句：{query_text[:120]}"),
        _step(
            "2.2",
            "UIE 抽实体（部位/疾病/症状）",
            _mentions_line(mentions) + f"；目录 {ms_to_sec_text(prep_timing.get('catalog_ms'))}",
        ),
        _step(
            "2.3",
            "预处理对齐（mention→标准名）",
            _align_line(alignments)
            + f"；对齐 {ms_to_sec_text(prep_timing.get('extract_align_ms'))}。"
            " 此处可走同义词 / Chroma 粗召回 + DeepSeek 精排，与下方工具 RAG 增强不是同一层。",
        ),
        _step(
            "2.4",
            "解析检索 seeds",
            numbered_lines(seeds[:12], empty="（无，仅用问句）") + "\n供工具拼主查询（见第三块序号列表）",
        ),
    ]

    enhance_steps: list[dict[str, str]] = []
    if not rag_on:
        plan = format_query_plan(seeds=seeds, query_text=query_text)
        enhance_steps.append(
            _step(
                "3.1",
                "RAG 增强开关 + 查询词",
                "当前关闭（config 或进程覆盖）。工具走基线单路召回，无多 query / RRF / 增强缓存。\n\n" + plan,
            )
        )
        if source == "neo4j" or channel == "neo4j":
            enhance_steps.append(
                _step("3.2", "Graph 基线", f"Neo4j Cypher → {len(packed.get('items') or [])} 条 items，无 enhance_graph_pack。")
            )
        elif channel == "ragflow":
            enhance_steps.append(_step("3.2", "RAGFlow 基线", "单句 retrieve，无 expand_queries / RRF。"))
        else:
            enhance_steps.append(_step("3.2", "Chroma 基线", "单路 _chroma_snips，无多 query / RRF。"))
    elif rag and rag.get("enhanced"):
        main_q = str(rag.get("main_query") or rag.get("cache_key") or query_text or "")
        queries = [str(q) for q in (rag.get("queries") or []) if str(q).strip()]
        plan = format_query_plan(
            seeds=seeds,
            query_text=query_text,
            main_query=main_q,
            expand_queries=queries,
            multi_query=cfg.multi_query,
            max_queries=cfg.max_queries,
        )
        expand_detail = numbered_lines(queries, empty="-") if queries else str(rag.get("query_expand_detail") or "-")
        enhance_steps.extend(
            [
                _step("3.1", "seeds + 主查询 + 待检索句", plan),
                _step(
                    "3.2",
                    "可选缓存",
                    str(rag.get("cache_detail") or "-"),
                ),
                _step(
                    "3.3",
                    "多查询扩展（规则，无 LLM）",
                    expand_detail if queries else str(rag.get("query_expand_detail") or "-"),
                ),
                _step(
                    "3.4",
                    "多路召回",
                    str(rag.get("recall_detail") or "-"),
                ),
                _step(
                    "3.5",
                    "RRF 融合",
                    str(rag.get("rrf_detail") or "-"),
                ),
                _step(
                    "3.6",
                    "关键词重排",
                    str(rag.get("rerank_detail") or "-"),
                ),
                _step(
                    "3.7",
                    "top_k 写回",
                    str(rag.get("output_detail") or "-"),
                ),
                _step(
                    "3.8",
                    "pack.rag 元数据",
                    f"channel={rag.get('channel')}；note={rag.get('note')}；"
                    f"管线 {ms_to_sec_text((rag.get('timing_ms') or {}).get('pipeline_ms'))}\n"
                    + (
                        "待检索句：\n" + numbered_lines(queries, empty="（无）")
                        if queries
                        else f"queries={rag.get('queries')}"
                    ),
                ),
            ]
        )
    elif source == "neo4j" or channel == "neo4j":
        plan = format_query_plan(seeds=seeds, query_text=query_text, main_query=query_text)
        enhance_steps.extend(
            [
                _step("3.1", "Graph 召回 + 查询词", plan),
                _step("3.2", "多 query / RRF", f"Neo4j 已返回 {len(packed.get('items') or [])} 条；图谱增强不做多 query/RRF，仅关键词重排 + top_k。"),
                _step(
                    "3.3",
                    "关键词重排 + top_k",
                    str(rag.get("rerank_detail") or f"top_k={cfg.top_k}，管线 {ms_to_sec_text((rag.get('timing_ms') or {}).get('pipeline_ms'))}"),
                ),
                _step("3.4", "pack.rag", f"note={rag.get('note')}；hits={len(rag.get('hits') or [])} 条"),
            ]
        )
    else:
        enhance_steps.append(_step("3.1", "RAG 增强", "已开启但未拿到 rag 元数据（可能管线异常或基线路径）。"))

    note_steps = [
        _step(
            "—",
            "与预处理对齐精排的区别",
            "预处理：口语 mention → 标准名（Chroma + DeepSeek）。"
            " RAG 增强：工具返回后对片段/图谱条目做多 query、RRF、关键词重排；两层不混。",
        ),
    ]

    return [
        {"title": "一、意图（进线）", "steps": intent_steps},
        {"title": "二、抽实体与对齐（预处理）", "steps": extract_steps},
        {"title": "三、RAG 增强 / 工具召回", "steps": enhance_steps},
        {"title": "说明", "steps": note_steps},
    ]


def format_companion_rag_block(packed: dict[str, Any], *, tool: str = "", query_text: str = "") -> str:
    """陪聊调试：与评测台第三块同一套 3.1–3.8 文案。"""
    rag = packed.get("rag") if isinstance(packed.get("rag"), dict) else {}
    seeds = [str(s).strip() for s in (packed.get("seeds") or []) if str(s).strip()]
    q = (query_text or str(rag.get("main_query") or "")).strip()
    cfg = get_rag_config()
    queries = [str(item).strip() for item in (rag.get("queries") or []) if str(item).strip()]
    main_q = str(rag.get("main_query") or rag.get("cache_key") or q)
    if rag.get("enhanced"):
        plan = format_query_plan(
            seeds=seeds,
            query_text=q or main_q,
            main_query=main_q,
            expand_queries=queries,
            multi_query=cfg.multi_query,
            max_queries=cfg.max_queries,
        )
        steps = [
            ("3.1", "seeds + 主查询 + 待检索句", plan),
            ("3.2", "可选缓存", str(rag.get("cache_detail") or "-")),
            ("3.3", "多查询扩展（规则，无 LLM）", numbered_lines(queries) if queries else str(rag.get("query_expand_detail") or "-")),
            ("3.4", "多路召回", str(rag.get("recall_detail") or "-")),
            ("3.5", "RRF 融合", str(rag.get("rrf_detail") or "-")),
            ("3.6", "关键词重排", str(rag.get("rerank_detail") or "-")),
            ("3.7", "top_k 写回", str(rag.get("output_detail") or "-")),
            (
                "3.8",
                "pack.rag 元数据",
                f"channel={rag.get('channel')}；note={rag.get('note')}；"
                f"管线 {ms_to_sec_text((rag.get('timing_ms') or {}).get('pipeline_ms'))}\n"
                + ("待检索句：\n" + numbered_lines(queries, empty="（无）")),
            ),
        ]
    else:
        source = str(packed.get("source") or "")
        steps = [
            ("3.1", "基线召回", f"source={source or '-'}；未走增强管线（无 expand / RRF）。"),
            ("3.2", "seeds", numbered_lines(seeds)),
        ]
    hits = rag.get("hits") if isinstance(rag.get("hits"), list) else []
    snippets = [str(row.get("text") or "").strip() for row in hits if isinstance(row, dict) and str(row.get("text") or "").strip()]
    if not snippets:
        snippets = [str(item).strip() for item in (packed.get("items") or []) if str(item).strip()]
    lines = [f"工具 {tool}".strip(), "RAG 增强 / 工具召回"]
    for no, title, detail in steps:
        lines.append(f"{no} {title}")
        lines.append(detail or "-")
    lines.append("知识片段")
    lines.append(numbered_lines(snippets[:8], empty="（无片段）"))
    return "\n".join(lines)
