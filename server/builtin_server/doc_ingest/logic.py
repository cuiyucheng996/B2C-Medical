"""文档入库：文件级过滤、类型标记，PDF 走抽字、版面、过滤、切块、落库。"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_BYTES = 15 * 1024 * 1024
_IMAGE_SUFFIX = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_TABLE_SUFFIX = {".xls", ".xlsx", ".csv"}


def ingest_document(*, filename: str, data: bytes, heading: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    name = Path(filename or "upload.bin").name
    gate = _file_gate(name, data)
    if not gate["ok"]:
        flow = _flow_early_stop(gate, step2=None, local=None, preprocess=None, ragflow=None, chunks=[])
        return _finish({"ok": False, "stage": "文件级过滤", **gate}, flow=flow, elapsed_s=time.perf_counter() - t0)
    if gate["doc_type"] != "pdf":
        step2 = _step2_branch(gate, entered=False)
        flow = _flow_early_stop(gate, step2=step2, local=None, preprocess=None, ragflow=None, chunks=[])
        return _finish(
            {
                "ok": True,
                "stage": "类型分流",
                "ingested": False,
                "reason": "图片和表格只做类型标记，不抽字入库",
                **gate,
            },
            flow=flow,
            elapsed_s=time.perf_counter() - t0,
        )
    body = _ingest_pdf(name, data, gate, heading=heading)
    elapsed = time.perf_counter() - t0
    body["elapsed_s"] = round(elapsed, 2)
    _patch_flow_elapsed(body.get("flow"), elapsed)
    return body


def _finish(payload: dict[str, Any], *, flow: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    out = dict(payload)
    out["flow"] = flow
    out["elapsed_s"] = round(elapsed_s, 2)
    _patch_flow_elapsed(flow, elapsed_s)
    return out


def _patch_flow_elapsed(flow: Any, elapsed_s: float) -> None:
    if not isinstance(flow, list) or not flow:
        return
    last = flow[-1]
    if not isinstance(last, dict) or last.get("step") != 7:
        return
    for m in list(last.get("metrics") or []):
        if isinstance(m, dict) and m.get("label") == "耗时":
            m["value"] = f"{round(elapsed_s, 1)}s"


def _file_gate(name: str, data: bytes) -> dict[str, Any]:
    reasons: list[str] = []
    if not data:
        reasons.append("空文件")
    if len(data) > _MAX_BYTES:
        reasons.append("超过 15MB")
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        doc_type = "pdf"
    elif suffix in _IMAGE_SUFFIX:
        doc_type = "image"
    elif suffix in _TABLE_SUFFIX:
        doc_type = "table"
    else:
        doc_type = "unknown"
        reasons.append("不是 PDF、图片或表格")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "filename": name,
        "doc_type": doc_type,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest() if data else "",
        "mark": {
            "table_file": doc_type == "table",
            "image_file": doc_type == "image",
            "scanned": None,
            "native_pdf": None,
        },
    }


def _resolve_pdf_engine() -> str:
    from configs.paths_b2c import ensure_b2c_runtime_path, ensure_uvgraph_root

    ensure_b2c_runtime_path()
    ensure_uvgraph_root()
    from src.configs.config_loader import get_yaml_config

    prefer = str(((get_yaml_config() or {}).get("doc_ingest") or {}).get("pdf_engine") or "auto").strip().lower()
    if prefer == "auto":
        try:
            import importlib.util

            if importlib.util.find_spec("parser.pdf_parser") is not None:
                return "deepdoc"
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
        from algorithm.pdf_kb import mineru_available

        return "mineru" if mineru_available() else "pypdf"
    if prefer in {"mineru", "pypdf", "deepdoc", "unstructured"}:
        if prefer == "mineru":
            from algorithm.pdf_kb import mineru_available

            if not mineru_available():
                raise RuntimeError("已配置 doc_ingest.pdf_engine=mineru，但未安装 mineru 包")
        return prefer
    return "pypdf"


def _ingest_pdf(name: str, data: bytes, gate: dict[str, Any], *, heading: dict[str, Any] | None = None) -> dict[str, Any]:
    from configs.paths_b2c import ensure_b2c_runtime_path

    ensure_b2c_runtime_path()
    from algorithm.ragflow_kb import ping_ragflow, replace_pdf

    engine = _resolve_pdf_engine()
    tmp = Path(tempfile.mkdtemp()) / name
    tmp.write_bytes(data)
    try:
        from src.knowledge.mineru_layout_filter import mineru_heading_override

        with mineru_heading_override(heading):
            packed = replace_pdf(tmp, replace_same_name=True, chunk_source="local", engine=engine)
    finally:
        tmp.unlink(missing_ok=True)

    local = packed.get("local") if isinstance(packed.get("local"), dict) else {}
    if str(local.get("source") or "") == "mineru":
        try:
            from server.builtin_server.doc_ingest.layout_images import export_mineru_layout_images

            export_mineru_layout_images(name)
        except Exception:
            logger.warning("doc_ingest layout image export failed", exc_info=True)
    preprocess = local.get("preprocess") if isinstance(local.get("preprocess"), dict) else {}
    native = int(preprocess.get("native_chars") or 0)
    ocr = int(preprocess.get("ocr_chars") or 0)
    mark = dict(gate["mark"])
    mark["native_pdf"] = native > 0 and native >= ocr
    mark["scanned"] = ocr > native
    chunks = _chunk_preview(name, limit=10)
    rf_ping = ping_ragflow()
    not_done = [
        "文档级完全相同与相似去重",
        "无实质、过短、过长无分段、过期、冲突",
        "空格归一化（如「A是 B的儿子」与「A是B的儿子」）",
        "递归分块、语义分块、父子块",
    ]
    flow = _build_flow_seven(
        gate=gate,
        entered_pdf=True,
        local=local,
        preprocess=preprocess,
        ragflow=packed.get("ragflow"),
        rf_ping=rf_ping,
        chunks=chunks,
        mark=mark,
        not_done=not_done,
    )
    return {
        "ok": True,
        "stage": "完成",
        "ingested": True,
        "pdf_engine": engine,
        "filename": name,
        "doc_type": "pdf",
        "size": gate["size"],
        "sha256": gate["sha256"],
        "mark": mark,
        "chunk_count": local.get("chunk_count"),
        "total_chunks": local.get("total_chunks"),
        "store": local.get("store"),
        "source": local.get("source"),
        "ragflow": packed.get("ragflow"),
        "native_chars": native,
        "ocr_chars": ocr,
        "preprocess_pages": preprocess.get("pages"),
        "not_done": not_done,
        "flow": flow,
    }


def _flow_early_stop(
    gate: dict[str, Any],
    *,
    step2: dict[str, Any] | None,
    local: Any,
    preprocess: Any,
    ragflow: Any,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    s2 = step2 or _step2_branch(gate, entered=gate.get("ok") and gate.get("doc_type") == "pdf")
    return _build_flow_seven(
        gate=gate,
        entered_pdf=False,
        local=local,
        preprocess=preprocess,
        ragflow=ragflow,
        rf_ping={},
        chunks=chunks,
        mark=gate.get("mark") or {},
    )


def _step1_gate(gate: dict[str, Any]) -> dict[str, Any]:
    ok = bool(gate.get("ok"))
    size_kb = round(int(gate.get("size") or 0) / 1024, 1)
    headline = "校验通过，可以进入下一步。" if ok else "校验未通过：" + "；".join(gate.get("reasons") or [])
    return {
        "step": 1,
        "name": "文件级",
        "status": "pass" if ok else "fail",
        "headline": headline,
        "metrics": [
            {"label": "文件名", "value": gate.get("filename")},
            {"label": "大小", "value": f"{size_kb} KB"},
            {"label": "识别类型", "value": gate.get("doc_type")},
        ],
        "tags": [
            {"text": "图片文件", "on": (gate.get("mark") or {}).get("image_file")},
            {"text": "表格文件", "on": (gate.get("mark") or {}).get("table_file")},
        ],
        "sha256_short": (gate.get("sha256") or "")[:16] + "…" if gate.get("sha256") else "—",
    }


def _step2_branch(gate: dict[str, Any], *, entered: bool) -> dict[str, Any]:
    dt = gate.get("doc_type")
    if dt == "pdf" and entered:
        return {
            "step": 2,
            "name": "分流",
            "status": "pass",
            "headline": "判定为 PDF，进入抽字、版面、切块、落库全链路。",
            "route": "pdf_full",
        }
    if dt in ("image", "table"):
        kind = "图片" if dt == "image" else "表格"
        return {
            "step": 2,
            "name": "分流",
            "status": "skip",
            "headline": f"判定为{kind}，仅打类型标记，不写入知识库。",
            "route": "mark_only",
        }
    return {
        "step": 2,
        "name": "分流",
        "status": "skip",
        "headline": "未进入 PDF 链路。",
        "route": "none",
    }


def _step3_parse(preprocess: dict[str, Any], mark: dict[str, Any]) -> dict[str, Any]:
    pages = int(preprocess.get("pages") or 0)
    native = int(preprocess.get("native_chars") or 0)
    ocr = int(preprocess.get("ocr_chars") or 0)
    ocr_pages = int(preprocess.get("ocr_pages") or 0)
    kinds = preprocess.get("block_kinds") if isinstance(preprocess.get("block_kinds"), dict) else {}
    kind_items = [{"name": k, "count": v} for k, v in sorted(kinds.items())]
    page_rows = list(preprocess.get("page_rows") or [])[:24]
    samples = list(preprocess.get("samples") or [])[:8]
    sub_stages = []
    for row in list(preprocess.get("stages") or []):
        if not isinstance(row, dict):
            continue
        sub_stages.append(
            {
                "title": str(row.get("title") or ""),
                "ok": bool(row.get("ok")),
                "detail": str(row.get("detail") or ""),
            }
        )
    doc_style = "偏电子版" if mark.get("native_pdf") else ("偏扫描" if mark.get("scanned") else "混合")
    return {
        "step": 3,
        "name": "PDF 解析",
        "status": "pass" if pages > 0 and sub_stages else "warn",
        "headline": f"共 {pages} 页；电子字 {native}、OCR 补字 {ocr}（{ocr_pages} 页）；整体{doc_style}。",
        "metrics": [
            {"label": "页数", "value": pages},
            {"label": "电子字", "value": native},
            {"label": "OCR 字", "value": ocr},
            {"label": "内容块类型", "value": len(kind_items)},
        ],
        "sub_stages": sub_stages,
        "block_kinds": kind_items,
        "page_rows": page_rows,
        "block_samples": samples,
    }


def _step4_chunk(chunks: list[dict[str, Any]], local: dict[str, Any]) -> dict[str, Any]:
    n = int(local.get("chunk_count") or 0)
    return {
        "step": 4,
        "name": "切块",
        "status": "pass" if n > 0 else "fail",
        "headline": f"按标题/目录截断，共切成 {n} 段；下方为前几段预览。",
        "rule": "MinerU 版面：每个 title/paragraph_title 与其后 text/table/image 合并为一块，直到下一标题；超长按 local_max_chars 再切。",
        "chunks": chunks,
    }


def _step5_store(local: dict[str, Any]) -> dict[str, Any]:
    n = int(local.get("chunk_count") or 0)
    total = int(local.get("total_chunks") or 0)
    store = str(local.get("store") or "")
    return {
        "step": 5,
        "name": "落库",
        "status": "pass" if n > 0 and store else "fail",
        "headline": f"同名 PDF 已整份替换；本文件 {n} 块，库内合计 {total} 块。",
        "metrics": [
            {"label": "存储文件", "value": store or "—"},
            {"label": "解析来源", "value": local.get("source") or "—"},
            {"label": "同名策略", "value": "整份替换"},
        ],
    }


def _step6_ragflow(ragflow: Any, rf_ping: dict[str, Any]) -> dict[str, Any]:
    if not rf_ping.get("ok"):
        return {
            "step": 6,
            "name": "同步 RAGFlow",
            "status": "skip",
            "headline": "RAGFlow 未连通或未配置，仅更新本地块库。",
            "detail": str(rf_ping.get("error") or "不可达"),
        }
    if not isinstance(ragflow, dict):
        return {
            "step": 6,
            "name": "同步 RAGFlow",
            "status": "warn",
            "headline": "RAGFlow 在线，但本次未拿到同步明细。",
        }
    if ragflow.get("ok") is False:
        return {
            "step": 6,
            "name": "同步 RAGFlow",
            "status": "warn",
            "headline": "本地已入库，但同步 RAGFlow 失败（常见：容器经代理调 DashScope 向量失败）。",
            "detail": str(ragflow.get("error") or "")[:400],
        }
    pushed = int(ragflow.get("chunk_count") or 0)
    return {
        "step": 6,
        "name": "同步 RAGFlow",
        "status": "pass" if ragflow.get("document_id") else "warn",
        "headline": f"已删同名文档并推送本地切块（chunk_source=local），写入 {pushed} 块。",
        "metrics": [
            {"label": "数据集", "value": ragflow.get("dataset_id")},
            {"label": "文档 ID", "value": ragflow.get("document_id")},
            {"label": "Web", "value": ragflow.get("web_url") or "—"},
        ],
    }


def _step7_summary(
    *,
    gate: dict[str, Any],
    ingested: bool,
    local: dict[str, Any] | None,
    preprocess: dict[str, Any] | None,
    elapsed_s: float,
    not_done: list[str],
) -> dict[str, Any]:
    prep = preprocess or {}
    loc = local or {}
    return {
        "step": 7,
        "name": "返回总览",
        "status": "pass" if ingested or gate.get("ok") else "fail",
        "headline": "入库完成，以下为关键数字与尚未实现的增强项。",
        "metrics": [
            {"label": "是否入库", "value": "是" if ingested else "否"},
            {"label": "块数", "value": loc.get("chunk_count") if ingested else "—"},
            {"label": "页数", "value": prep.get("pages") if ingested else "—"},
            {"label": "耗时", "value": f"{elapsed_s:.1f}s"},
        ],
        "not_done": not_done,
    }


def _build_flow_seven(
    *,
    gate: dict[str, Any],
    entered_pdf: bool,
    local: dict[str, Any] | None,
    preprocess: dict[str, Any] | None,
    ragflow: Any,
    rf_ping: dict[str, Any],
    chunks: list[dict[str, Any]],
    mark: dict[str, Any],
    not_done: list[str] | None = None,
) -> list[dict[str, Any]]:
    prep = preprocess if isinstance(preprocess, dict) else {}
    loc = local if isinstance(local, dict) else {}
    ingested = entered_pdf and bool(loc.get("chunk_count"))
    nd = not_done or [
        "文档级完全相同与相似去重",
        "低质段落与过期/冲突信息过滤",
        "空格归一化",
        "递归 / 语义 / 父子块",
    ]
    steps: list[dict[str, Any]] = [_step1_gate(gate), _step2_branch(gate, entered=entered_pdf)]
    if entered_pdf and gate.get("ok"):
        steps.append(_step3_parse(prep, mark))
        steps.append(_step4_chunk(chunks, loc))
        steps.append(_step5_store(loc))
        steps.append(_step6_ragflow(ragflow, rf_ping))
    else:
        for num, name in ((3, "PDF 解析"), (4, "切块"), (5, "落库"), (6, "同步 RAGFlow")):
            steps.append(
                {
                    "step": num,
                    "name": name,
                    "status": "skip",
                    "headline": "本文件未进入 PDF 入库链路，此步跳过。",
                }
            )
    steps.append(
        _step7_summary(
            gate=gate,
            ingested=ingested,
            local=loc if ingested else None,
            preprocess=prep if ingested else None,
            elapsed_s=0,
            not_done=nd if (ingested or gate.get("doc_type") == "pdf") else [],
        )
    )
    return steps


def _chunk_preview(doc_name: str, *, limit: int) -> list[dict[str, Any]]:
    from algorithm.pdf_kb import load_local_chunks

    rows = [r for r in load_local_chunks() if str(r.get("doc") or "") == doc_name]
    rows.sort(key=lambda item: int(item.get("ord") or 0))
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        text = str(row.get("content") or "").strip()
        out.append(
            {
                "ord": int(row.get("ord") or 0) + 1,
                "page": row.get("page"),
                "chars": len(text),
                "preview": text[:320] + ("…" if len(text) > 320 else ""),
            }
        )
    return out


def health_payload() -> dict[str, str]:
    return {"status": "healthy", "service": "doc_ingest"}
