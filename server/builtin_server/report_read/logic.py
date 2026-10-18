"""报告解读业务：闸门、抽指标、查手册、拼结果。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from server.builtin_server.report_read.gate import ingest_user_file
from server.builtin_server.report_read.handbook import lookup_report_handbook
from server.builtin_server.report_read.ingest import _extract_indicators, _pdf_extract_module, mineru_split_tables_cached

__all__ = [
    "collect_report_facts",
    "compose_report",
    "gate_report_file",
    "lookup_report_handbook",
    "pull_report_facts",
]


def gate_report_file(
    *,
    filename: str,
    data: bytes,
    source: str = "api",
    user_id: str = "report-api",
) -> dict[str, Any]:
    src = "tool" if str(source or "").strip().lower() == "tool" else "api"
    ingested = ingest_user_file(user_id=user_id or "report-api", filename=filename, data=data)
    compliant = ingested.role == "lab_report"
    return {
        "ok": src == "tool" or compliant,
        "source": src,
        "skipped_gate": src == "tool",
        "compliant": compliant,
        "role": ingested.role,
        "reasons": list(ingested.reasons),
        "score": ingested.score_card,
        "last_file": ingested.to_dict(),
    }


def collect_report_facts(
    *,
    last_report: dict[str, Any] | None = None,
    last_file: dict[str, Any] | None = None,
    layout: str = "v2",
) -> dict[str, Any]:
    """从已入库文件抽出标题、表、指标。不含手册检索。"""
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
            table_json, pulled, mineru_source = mineru_split_tables_cached(pdf_path, file_row, layout=layout)
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
    empty = not filename and not indicators and not chunks and not tables and not table_json
    names = [str(item.get("name") or "") for item in indicators[:12] if item.get("name")]
    uid = str(file_row.get("user_id") or (chunks[0].get("user_id") if chunks else "") or "")
    out_report = {
        "filename": filename,
        "saved_path": saved,
        "mineru_pdf_path": mineru_pdf,
        "user_id": uid,
        "parse_source": parse_source or "preprocess_layout",
        "titles": titles[:8],
        "indicators": indicators,
        "tables": table_json,
        "table_count": len(table_json) or table_count,
        "chunk_count": len(chunks),
        "chunks": chunks[:40],
        "markdown_preview": str(file_row.get("markdown_preview") or report.get("markdown_preview") or "")[:800],
        "user_text": str(file_row.get("user_text") or report.get("user_text") or "")[:2000],
    }
    return {
        "empty": empty,
        "filename": filename,
        "indicator_names": names,
        "table_json_count": len(table_json),
        "table_count": len(table_json) or table_count,
        "indicator_count": len(indicators),
        "report": out_report,
    }


def pull_report_facts(last_file: dict[str, Any] | None, layout: str = "v2") -> dict[str, Any]:
    return collect_report_facts(
        last_file=last_file if isinstance(last_file, dict) else {},
        layout=layout,
    )


def compose_report(
    *,
    gate: dict[str, Any] | None,
    facts: dict[str, Any] | None,
    handbook: dict[str, Any] | None,
) -> dict[str, Any]:
    gate = gate if isinstance(gate, dict) else {}
    facts = facts if isinstance(facts, dict) else {}
    handbook = handbook if isinstance(handbook, dict) else {}
    items = list(handbook.get("items") or [])
    bits = [f"已解读 {facts.get('filename') or '报告'}"]
    if facts.get("table_json_count"):
        bits.append(f"结构化表 {facts.get('table_json_count')} 张")
    elif facts.get("table_count"):
        bits.append(f"表区域 {facts.get('table_count')} 处")
    bits.append(f"抽出 {facts.get('indicator_count') or 0} 条指标")
    if items:
        bits.append(f"手册命中 {len(items)} 条")
    return {
        "stopped": False,
        "source": gate.get("source"),
        "skipped_gate": gate.get("skipped_gate"),
        "compliant": gate.get("compliant"),
        "role": gate.get("role"),
        "reasons": gate.get("reasons") or [],
        "score": gate.get("score") or {},
        "summary": "，".join(bits),
        "report": facts.get("report") or {},
        "items": items[:16],
        "kb_source": handbook.get("kb_source") or "",
        "handbook_query": handbook.get("query") or "",
    }
