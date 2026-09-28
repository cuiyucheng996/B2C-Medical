"""报告指标解读三件事：上传解读、口头提报告、指标对历史。"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "compare_lab_history",
    "explain_mentioned_report",
    "explain_uploaded_report",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def _mentions_list(mentions: dict[str, Any] | None, key: str) -> list[str]:
    row = mentions if isinstance(mentions, dict) else {}
    return [str(item).strip() for item in (row.get(key) or []) if str(item).strip()]


def _handbook(query: str, handler: str) -> dict[str, Any]:
    from algorithm.kg_query import search_chroma_knowledge, search_exam_handbook

    packed = search_exam_handbook(query=query, handler=handler)
    items = list(packed.get("items") or [])
    if items:
        return packed
    extra = search_chroma_knowledge(query=query, handler=handler)
    return extra if extra.get("items") else packed


def _facts(user_id: str, query: str) -> list[dict[str, Any]]:
    uid = (user_id or "").strip()
    q = (query or "").strip()
    if not uid or not q:
        return []
    from memory.graphiti_store import search_facts

    return list(search_facts(user_id=uid, query=q) or [])


def _match_indicators(indicators: list[dict[str, Any]], metric: str) -> list[dict[str, str]]:
    needle = _norm(metric)
    if not needle:
        return []
    hits: list[dict[str, str]] = []
    for item in indicators or []:
        name = str(item.get("name") or "")
        if needle in _norm(name) or _norm(name) in needle:
            hits.append(
                {
                    "name": name,
                    "value": str(item.get("value") or ""),
                    "unit": str(item.get("unit") or ""),
                    "flag": str(item.get("flag") or ""),
                }
            )
    return hits[:8]


def explain_uploaded_report(
    *,
    query: str = "",
    last_report: dict[str, Any] | None = None,
    last_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """预处理判定报告解读且给了 PDF：MinerU 拆该 PDF；否则只用切块抽指标。"""
    from pathlib import Path

    from algorithm.report_ingest import _extract_indicators, _pdf_extract_module, mineru_split_tables

    report = last_report if isinstance(last_report, dict) else {}
    file_row = last_file if isinstance(last_file, dict) else {}
    chunks = list(file_row.get("chunks") or report.get("chunks") or [])
    titles = list(file_row.get("titles") or report.get("titles") or [])
    tables = list(file_row.get("tables") or report.get("tables") or [])
    body = str(file_row.get("body") or "")
    parse_source = str(file_row.get("parse_source") or report.get("parse_source") or "")
    filename = str(file_row.get("filename") or report.get("filename") or "")
    saved = str(file_row.get("saved_path") or report.get("saved_path") or "")
    table_count = int(file_row.get("table_count") or report.get("table_count") or 0)
    if not tables:
        tables = [
            str(item.get("html") or item.get("text") or "")
            for item in chunks
            if str(item.get("category") or "") == "Table"
        ]
        tables = [item for item in tables if item]
    if not table_count:
        table_count = sum(1 for item in chunks if str(item.get("category") or "") == "Table") or len(tables)
    if not titles:
        titles = [
            str(item.get("text") or "")
            for item in chunks
            if str(item.get("category") or "") in {"Title", "Header", "Headline"}
        ]
        titles = [item for item in titles if item]
    if not body.strip():
        body = "\n".join(
            str(item.get("text") or "")
            for item in chunks
            if str(item.get("category") or "") not in {"Title", "Header", "Headline", "Table", "Image", "Figure", "Picture"}
            and str(item.get("text") or "").strip()
        )
    table_json: list[dict] = []
    indicators = list(file_row.get("indicators") or report.get("indicators") or [])
    mineru_pdf = str(file_row.get("mineru_pdf_path") or report.get("mineru_pdf_path") or "")
    pdf_path = Path(mineru_pdf) if mineru_pdf else None
    if pdf_path is not None and pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
        try:
            table_json, pulled, mineru_source = mineru_split_tables(pdf_path)
            if pulled:
                indicators = pulled
            if mineru_source:
                parse_source = mineru_source
        except Exception:
            table_json = []
    if not table_json and tables:
        to_json = getattr(_pdf_extract_module(), "table_to_json", None)
        if to_json:
            table_json = [to_json(item) for item in tables]
        if not indicators:
            indicators = _extract_indicators(tables, body)
    if not filename and not indicators and not chunks and not tables and not table_json:
        return {
            "summary": "本轮没有上传检查报告",
            "report": {},
            "items": [],
            "empty": True,
        }
    names = [str(item.get("name") or "") for item in indicators[:12] if item.get("name")]
    q = (query or "").strip() or " ".join(names[:8]) or filename or "化验单"
    kb = _handbook(q[:200], "report_upload")
    items = list(kb.get("items") or [])
    bits = [f"已解读 {filename or '报告'}"]
    if table_json:
        bits.append(f"结构化表 {len(table_json)} 张")
    elif table_count:
        bits.append(f"表区域 {table_count} 处")
    bits.append(f"抽出 {len(indicators)} 条指标")
    if items:
        bits.append(f"手册命中 {len(items)} 条")
    uid = str(file_row.get("user_id") or (chunks[0].get("user_id") if chunks else "") or "")
    out_report = {
        "filename": filename,
        "saved_path": saved,
        "mineru_pdf_path": mineru_pdf,
        "user_id": uid,
        "parse_source": parse_source or "preprocess_layout",
        "titles": titles[:8],
        "indicators": indicators[:25],
        "tables": table_json[:8],
        "table_count": len(table_json) or table_count,
        "chunk_count": len(chunks),
        "chunks": chunks[:40],
        "markdown_preview": str(file_row.get("markdown_preview") or report.get("markdown_preview") or "")[:800],
        "user_text": str(file_row.get("user_text") or report.get("user_text") or "")[:2000],
    }
    return {
        "summary": "，".join(bits),
        "report": out_report,
        "items": items[:16],
        "kb_source": kb.get("source") or "",
        "empty": False,
    }


def explain_mentioned_report(
    *,
    exam: str = "",
    query: str = "",
    user_id: str = "",
    mentions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用户口头提到的检查/化验单，查记忆 + 手册。"""
    exam_text = (exam or "").strip() or " ".join(_mentions_list(mentions, "检查"))
    q = " ".join(part for part in (exam_text, (query or "").strip()) if part).strip() or "检查报告"
    facts = _facts(user_id, q)
    kb = _handbook(q[:200], "report_explain")
    items = list(kb.get("items") or [])
    bits = []
    if exam_text:
        bits.append(f"检查：{exam_text}")
    if facts:
        bits.append(f"记忆 {len(facts)} 条")
    if items:
        bits.append(f"手册 {len(items)} 条")
    summary = "，".join(bits) or "没有找到这份检查的记忆或手册"
    return {
        "summary": summary,
        "exam": exam_text,
        "facts": facts[:12],
        "items": items[:16],
        "kb_source": kb.get("source") or "",
        "empty": not facts and not items,
    }


def compare_lab_history(
    *,
    metric: str = "",
    query: str = "",
    user_id: str = "",
    last_report: dict[str, Any] | None = None,
    mentions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """当前报告里的某指标 vs 长期记忆里同名记录。"""
    metric_text = (metric or "").strip() or " ".join(_mentions_list(mentions, "指标"))
    if not metric_text:
        # 从「对比一下胆固醇」这类话里抠后半截当指标名
        text = re.sub(r"(对比|比较|和上次|跟上次|历史|变化)", "", query or "")
        text = re.sub(r"(一下|看看|的|了)", "", text).strip()
        metric_text = text[:20]
    report = last_report if isinstance(last_report, dict) else {}
    current = _match_indicators(list(report.get("indicators") or []), metric_text)
    facts = _facts(user_id, metric_text or (query or ""))
    kb = _handbook((metric_text or query or "")[:200], "report_compare")
    items = list(kb.get("items") or [])
    if not metric_text:
        return {
            "summary": "没有指出要对比哪项指标",
            "metric": "",
            "current": [],
            "facts": [],
            "items": [],
            "empty": True,
            "need_metric": True,
        }
    bits = [f"指标：{metric_text}"]
    if current:
        bits.append(f"本轮 {len(current)} 条")
    if facts:
        bits.append(f"历史 {len(facts)} 条")
    if items:
        bits.append("含参考范围")
    return {
        "summary": "，".join(bits),
        "metric": metric_text,
        "current": current,
        "facts": facts[:12],
        "items": items[:12],
        "kb_source": kb.get("source") or "",
        "empty": not current and not facts,
        "need_metric": False,
    }
