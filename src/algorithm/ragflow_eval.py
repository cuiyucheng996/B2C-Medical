"""评测：手册 PDF 本地版面分块 → 写入 RAGFlow（与 replace_pdf 同一条业务链）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from configs.paths_b2c import B2C_ROOT
from algorithm.pdf_kb import ingest_local_pdf, load_local_chunks
from algorithm.ragflow_kb import ping_ragflow, replace_pdf

__all__ = ["run_ragflow_ingest_eval"]

_EVAL_DIR = B2C_ROOT / "data" / "eval_ingest"
_MAX_BYTES = 25 * 1024 * 1024


def _method_label(source: str) -> str:
    if source == "ragflow":
        return "RAGFlow naive"
    return "DeepDoc title 分块"


def _step(sid: str, title: str, ok: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"id": sid, "title": title, "ok": ok, "detail": detail}
    if extra:
        row["extra"] = extra
    return row


def run_ragflow_ingest_eval(
    *,
    filename: str,
    data: bytes,
    chunk_source: str = "local",
    skip_ragflow: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    name = Path(filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        steps.append(_step("file", "接收文件", False, "只接受 PDF（手册入库链路）"))
        return {"ok": False, "steps": steps, "previews": []}
    if not data:
        steps.append(_step("file", "接收文件", False, "文件为空"))
        return {"ok": False, "steps": steps, "previews": []}
    if len(data) > _MAX_BYTES:
        steps.append(_step("file", "接收文件", False, f"超过 {_MAX_BYTES} 字节"))
        return {"ok": False, "steps": steps, "previews": []}

    safe = re.sub(r"[^\w.\u4e00-\u9fff-]+", "_", name)
    eval_name = safe if safe.startswith("eval__") else f"eval__{safe}"
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    dest = _EVAL_DIR / eval_name
    dest.write_bytes(data)
    steps.append(_step("file", "保存上传件", True, f"{eval_name} · {len(data)} 字节"))

    ping = ping_ragflow()
    ping_ok = bool(ping.get("ok"))
    steps.append(
        _step(
            "ping",
            "探测 RAGFlow",
            ping_ok,
            str(ping.get("detail") or ping.get("error") or ("通" if ping_ok else "未通")),
        )
    )

    source = (chunk_source or "local").strip().lower()
    if source not in {"local", "ragflow"}:
        source = "local"
    try:
        if skip_ragflow:
            packed = {
                "local": ingest_local_pdf(dest, replace_same_name=True, with_tsr=False),
                "ragflow": None,
            }
        else:
            packed = replace_pdf(dest, replace_same_name=True, chunk_source=source)
    except Exception as exc:
        steps.append(_step("ingest", "本地解析并入库", False, str(exc)))
        return {"ok": False, "steps": steps, "previews": [], "ping": ping}

    local = packed.get("local") or {}
    local_n = int(local.get("chunk_count") or 0)
    preprocess = local.get("preprocess") if isinstance(local.get("preprocess"), dict) else {}
    phases = list(preprocess.get("phases") or [])
    for phase in phases:
        steps.append(
            _step(
                str(phase.get("id") or "pre"),
                str(phase.get("title") or ""),
                True,
                str(phase.get("aim") or ""),
                extra={
                    "phase": phase,
                    "branches": list(phase.get("branches") or []),
                    "facts": list(phase.get("facts") or []),
                },
            )
        )
    for stage in preprocess.get("stages") or []:
        steps.append(
            _step(
                str(stage.get("id") or "pre"),
                "分块前 · " + str(stage.get("title") or ""),
                bool(stage.get("ok")),
                str(stage.get("detail") or ""),
                extra={"ops": list(stage.get("ops") or [])},
            )
        )
    steps.append(
        _step(
            "layout",
            "本地版面分块",
            local_n > 0,
            f"source={local.get('source') or '-'} · 本文件 {local_n} 块 · 库内共 {local.get('total_chunks') or 0}",
            extra={"store": local.get("store") or ""},
        )
    )
    previews = []
    for row in load_local_chunks():
        if str(row.get("doc") or "") != eval_name:
            continue
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        previews.append(
            {
                "ord": row.get("ord"),
                "page": row.get("page"),
                "text": text[:280],
            }
        )
        if len(previews) >= 8:
            break
    steps.append(
        _step(
            "preview",
            "分块预览",
            bool(previews),
            f"展示前 {len(previews)} 块",
        )
    )

    remote = packed.get("ragflow")
    if not remote:
        steps.append(
            _step(
                "ragflow",
                "写入 RAGFlow",
                False,
                (
                    "按评测选项跳过（只跑本地分块，不动 RAGFlow 库）。本地块已落 exam_kb。"
                    if skip_ragflow
                    else "未写入（服务未通或跳过）。本地块已落 exam_kb。"
                ),
            )
        )
        return {
            "ok": local_n > 0,
            "local_only": True,
            "skip_ragflow": bool(skip_ragflow),
            "steps": steps,
            "previews": previews,
            "block_samples": list(preprocess.get("samples") or []),
            "page_rows": list(preprocess.get("page_rows") or []),
            "phases": phases,
            "preprocess": preprocess,
            "method": _method_label(source),
            "ping": ping,
            "local": local,
            "ragflow": None,
        }

    pushed = int(remote.get("chunk_count") or (remote.get("parse") or {}).get("chunk_count") or 0)
    steps.append(
        _step(
            "ragflow",
            "写入 RAGFlow",
            True,
            f"chunk_source={remote.get('chunk_source')} · 文档 {remote.get('name')} · 块 {pushed or '解析中/见 Web'}",
            extra={
                "dataset_id": remote.get("dataset_id") or "",
                "document_id": remote.get("document_id") or "",
                "web_url": remote.get("web_url") or "",
            },
        )
    )
    return {
        "ok": True,
        "local_only": False,
        "steps": steps,
        "previews": previews,
        "block_samples": list(preprocess.get("samples") or []),
        "page_rows": list(preprocess.get("page_rows") or []),
        "phases": phases,
        "preprocess": preprocess,
        "method": _method_label(source),
        "ping": ping,
        "local": local,
        "ragflow": remote,
    }
