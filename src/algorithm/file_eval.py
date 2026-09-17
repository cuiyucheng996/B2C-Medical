"""评测：复刻上传闸门 → 打分锁大类 → 转 PDF / 结构化块 → 报告解读拆表。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algorithm.file_gate import ingest_user_file, intent_for_file_role, score_upload_role
from algorithm.report_read import explain_uploaded_report
from algorithm.unstructured_tables import extract_tables_unstructured
from tools.companion_tools import tools_for_intent

__all__ = ["run_file_eval"]

_TABLE_MEDIA = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def run_file_eval(
    *,
    filename: str,
    data: bytes,
    user_id: str = "eval-file",
    table_engine: str = "gate",
    us_strategy: str = "hi_res",
    us_infer_table: bool = True,
    us_languages: str = "eng,chi_sim",
    gate_lab_threshold: int = 2,
    gate_run_explain: bool = True,
    gate_layout_only: bool = True,
) -> dict[str, Any]:
    name = (filename or "upload.bin").strip() or "upload.bin"
    mode = (table_engine or "gate").strip().lower()
    if mode not in {"gate", "unstructured"}:
        mode = "gate"
    ingested = ingest_user_file(
        user_id=user_id or "eval-file",
        filename=name,
        data=data,
        layout_only=bool(gate_layout_only) if mode == "gate" else True,
        lab_threshold=max(1, int(gate_lab_threshold or 2)),
    )
    card = dict(ingested.score_card or {})
    if not card:
        card = score_upload_role(
            family=ingested.family,
            table_count=ingested.table_count,
            indicator_count=len(ingested.indicators or []),
            text="\n".join([ingested.filename, *(ingested.titles or []), ingested.body or ""]),
        )
    spec = intent_for_file_role(ingested.role) or {}
    tools = [
        str(getattr(item, "name", "") or "")
        for item in tools_for_intent(major=str(spec.get("major") or ""), handler=str(spec.get("handler") or ""))
    ]
    pdf_path = str(ingested.mineru_pdf_path or "")
    if pdf_path:
        artifact = "pdf"
    elif ingested.chunks or ingested.tables or ingested.body:
        artifact = "structured_chunks"
    else:
        artifact = "empty"
    split: dict[str, Any] | None = None
    if ingested.role == "lab_report" and mode == "gate" and gate_run_explain:
        split = explain_uploaded_report(query=ingested.filename, last_file=ingested.to_dict())

    tables_pack: dict[str, Any] = {}
    preprocess_view: dict[str, Any] = {}
    block_samples: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    if mode == "unstructured":
        table_path = str(ingested.mineru_pdf_path or ingested.saved_path or "")
        suffix = Path(table_path).suffix.lower() if table_path else ""
        if suffix in _TABLE_MEDIA:
            langs = [p.strip() for p in (us_languages or "eng,chi_sim").replace("，", ",").split(",") if p.strip()]
            try:
                tables_pack = extract_tables_unstructured(
                    table_path,
                    strategy=us_strategy,
                    infer_table_structure=bool(us_infer_table),
                    languages=langs,
                )
            except Exception as exc:
                tables_pack = {"ok": False, "error": str(exc), "preprocess": {}, "tables": []}
        else:
            tables_pack = {
                "ok": False,
                "error": "新方式仅支持 PDF / 图片（Unstructured hi_res）",
                "preprocess": {},
                "tables": [],
            }
        preprocess_view = tables_pack.get("preprocess") if isinstance(tables_pack.get("preprocess"), dict) else {}
        phases = list(preprocess_view.get("phases") or [])
        page_rows = list(preprocess_view.get("page_rows") or [])
        block_samples = list(preprocess_view.get("samples") or [])
        steps = list(preprocess_view.get("stages") or [])
        if ingested.role == "lab_report":
            split = {
                "summary": f"Unstructured 抽到 {tables_pack.get('table_count') or 0} 张表（评测未走 MinerU）",
                "report": {"tables": list(tables_pack.get("tables") or [])[:8]},
                "items": [],
            }
    else:
        for item in list(ingested.to_dict().get("chunks") or [])[:12]:
            block_samples.append(
                {
                    "type": str(item.get("category") or "Unknown"),
                    "page": item.get("page"),
                    "text": str(item.get("text") or item.get("html") or "")[:120],
                }
            )

    gate_options: dict[str, Any] = {}
    if mode == "gate":
        gate_options = {
            "lab_threshold": max(1, int(gate_lab_threshold or 2)),
            "run_explain": bool(gate_run_explain),
            "layout_only": bool(gate_layout_only),
            "strategy": "hi_res",
            "languages": ["eng", "chi_sim"],
            "worker": "pdf_layout_worker（子进程 UnstructuredPDFLoader / partition_image）",
        }

    return {
        "ok": True,
        "table_engine": mode,
        "filename": ingested.filename,
        "family": ingested.family,
        "role": ingested.role,
        "score_card": card,
        "intent": {
            "major": spec.get("major") or "",
            "intent": spec.get("intent") or "",
            "handler": spec.get("handler") or "",
        },
        "tools": tools,
        "preprocess": {
            "artifact": artifact,
            "saved_path": ingested.saved_path,
            "mineru_pdf_path": pdf_path,
            "reasons": list(ingested.reasons or []),
            "titles": list(ingested.titles or [])[:12],
            "table_count": ingested.table_count,
            "image_count": ingested.image_count,
            "chunks": list(ingested.to_dict().get("chunks") or []),
            "tables": list(ingested.tables or [])[:8],
            "body": (ingested.body or "")[:1500],
        },
        "report_split": split,
        "table_extract": {
            "ok": bool(tables_pack.get("ok")),
            "table_count": tables_pack.get("table_count") or 0,
            "tables": list(tables_pack.get("tables") or [])[:8],
            "error": tables_pack.get("error") or "",
            "loader_options": tables_pack.get("loader_options") or {},
        },
        "phases": phases,
        "page_rows": page_rows,
        "block_samples": block_samples,
        "steps": steps,
        "gate_options": gate_options,
    }
