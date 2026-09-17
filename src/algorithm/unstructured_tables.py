"""报告表格：UnstructuredPDFLoader hi_res（YOLOX 目标检测 + OCR + infer_table_structure）。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "describe_unstructured_preprocess",
    "elements_to_chunk_records",
    "extract_tables_unstructured",
]

_IMAGE_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def extract_tables_unstructured(
    path: str | Path,
    *,
    layout_only: bool = False,
    strategy: str = "hi_res",
    infer_table_structure: bool | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """PDF 走 UnstructuredPDFLoader；图片走 partition_image。表格出 HTML 保行列。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    infer = infer_table_structure if infer_table_structure is not None else (not layout_only)
    langs = [str(x).strip() for x in (languages or ["eng", "chi_sim"]) if str(x).strip()] or ["eng", "chi_sim"]
    strat = (strategy or "hi_res").strip().lower()
    if strat not in {"hi_res", "fast", "ocr_only", "auto"}:
        strat = "hi_res"
    opts = {
        "strategy": strat,
        "infer_table_structure": infer,
        "languages": langs,
        "mode": "elements",
    }
    packed = _run_layout(
        file_path,
        strategy=strat,
        infer_table_structure=infer,
        languages=langs,
    )
    elements = list(packed.get("elements") or [])
    preprocess = describe_unstructured_preprocess(elements, path=file_path, error=str(packed.get("error") or ""))
    tables = [
        {
            "page": item.get("page") or 1,
            "html": str(item.get("html") or ""),
            "text": str(item.get("text") or "")[:400],
        }
        for item in elements
        if str(item.get("category") or "") == "Table" or "<table" in str(item.get("html") or "").lower()
    ]
    records = elements_to_chunk_records(elements)
    return {
        "ok": bool(packed.get("ok")) and bool(elements),
        "error": packed.get("error") or "",
        "path": str(file_path),
        "elements": elements,
        "tables": tables,
        "table_count": len(tables),
        "records": records,
        "preprocess": preprocess,
        "loader_options": opts,
    }


def describe_unstructured_preprocess(
    elements: list[dict[str, Any]],
    *,
    path: Path | None = None,
    error: str = "",
) -> dict[str, Any]:
    cats = Counter(str(item.get("category") or "Unknown") for item in elements)
    pages = sorted({int(item.get("page") or 1) for item in elements}) if elements else []
    html_n = sum(1 for item in elements if "<table" in str(item.get("html") or "").lower())
    page_rows = []
    by_page: dict[int, list[dict[str, Any]]] = {}
    for item in elements:
        no = int(item.get("page") or 1)
        by_page.setdefault(no, []).append(item)
    for no in pages:
        rows = by_page.get(no) or []
        page_rows.append(
            {
                "page": no,
                "chars": sum(len(str(r.get("text") or "")) for r in rows),
                "native": 0,
                "ocr": sum(len(str(r.get("text") or "")) for r in rows),
                "backend": "unstructured_hi_res",
                "regions": len(rows),
                "lines": len(rows),
            }
        )
    label_txt = "、".join(f"{k}:{v}" for k, v in cats.most_common()) or "无"
    samples = []
    for item in elements:
        if len(samples) >= 12:
            break
        text = str(item.get("html") or item.get("text") or "")[:120]
        samples.append({"type": str(item.get("category") or "Unknown"), "page": item.get("page"), "text": text})
    phases = [
        {
            "id": "pages",
            "title": "① 按页转图 + 识别",
            "aim": "Poppler 把 PDF 渲成页图；图片直接进分区。字少走 OCR，字多走电子字。",
            "branches": [
                {
                    "cond": "字多（电子版）",
                    "handler": "pdfplumber / 内嵌文字层",
                    "note": "Unstructured 抽字符坐标，再聚成行",
                },
                {
                    "cond": "字太少（扫描页 / 图片）",
                    "handler": "OCR（Tesseract，chi_sim+eng）",
                    "note": "先 DeepDoc OCR 路径由 Unstructured 调度，不行再本机 Tesseract",
                },
                {
                    "cond": "两路都有",
                    "handler": "重叠去重",
                    "note": "重叠框优先留电子字，避免双份",
                },
            ],
            "facts": [
                ["文件", path.name if path is not None else "-"],
                ["元素数", str(len(elements))],
                ["页", str(len(pages) or 0)],
                ["错误", error or "无"],
            ],
        },
        {
            "id": "layout",
            "title": "② 版面识别 + 表格结构",
            "aim": "YOLOX 在页图上找标题/正文/表/图；表再 infer_table_structure 拆行列成 HTML。",
            "branches": [
                {
                    "cond": "任意页",
                    "handler": "YOLOX 目标检测（hi_res）",
                    "note": "框出 title / text / table / figure，比 PyPDFLoader 保结构",
                },
                {
                    "cond": "检出 Table",
                    "handler": "infer_table_structure",
                    "note": "单元格行列写入 text_as_html，避免只剩一坨字",
                },
            ],
            "facts": [
                ["类别", label_txt],
                ["带 HTML 的表", str(html_n)],
                ["策略", "hi_res + infer_table_structure"],
            ],
        },
    ]
    return {
        "stages": [
            {
                "id": "raster",
                "title": "转图 + OCR",
                "ok": bool(elements),
                "detail": f"{len(pages)} 页 · 元素 {len(elements)}",
                "ops": [
                    "Poppler 按页转图",
                    "电子字优先，扫描页 OCR",
                    "chi_sim + eng",
                ],
            },
            {
                "id": "layout",
                "title": "YOLOX + 表结构",
                "ok": bool(elements),
                "detail": f"{label_txt} · HTML 表 {html_n}",
                "ops": [
                    "hi_res：YOLOX 找标题/正文/表/图",
                    "infer_table_structure：表拆成 HTML 行列",
                ],
            },
        ],
        "phases": phases,
        "page_rows": page_rows,
        "samples": samples,
        "block_kinds": dict(cats),
        "pages": len(pages),
        "chars": sum(int(r.get("chars") or 0) for r in page_rows),
        "native_chars": 0,
        "ocr_chars": sum(int(r.get("ocr") or 0) for r in page_rows),
        "ocr_pages": len(pages),
        "region_labels": dict(cats),
        "line_boxes": len(elements),
        "tsr": True,
        "text_concat": False,
    }


def elements_to_chunk_records(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Title 切开；Table 单独成块（优先 HTML）。"""
    records: list[dict[str, Any]] = []
    buf: list[str] = []
    page = 1

    def flush() -> None:
        text = "\n".join(item for item in buf if item).strip()
        buf.clear()
        if text:
            records.append({"content": text, "page": page, "bbox": [0, len(records) * 20, 100, len(records) * 20 + 10]})

    for item in elements:
        page = int(item.get("page") or 1)
        cat = str(item.get("category") or "")
        html = str(item.get("html") or "").strip()
        text = str(item.get("text") or "").strip()
        body = html if html and ("<table" in html.lower() or cat == "Table") else text
        if not body:
            continue
        if cat == "Table" or (html and "<table" in html.lower()):
            flush()
            records.append(
                {
                    "content": body,
                    "page": page,
                    "bbox": [0, len(records) * 20, 100, len(records) * 20 + 10],
                }
            )
            continue
        if cat in {"Title", "Header", "Headline"}:
            flush()
        buf.append(body)
    flush()
    return records


def _run_layout(
    path: Path,
    *,
    strategy: str = "hi_res",
    infer_table_structure: bool = True,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    suffix = path.suffix.lower()
    langs = languages or ["eng", "chi_sim"]
    try:
        if suffix in _IMAGE_EXT:
            from unstructured.partition.image import partition_image

            parts = partition_image(
                filename=str(path),
                strategy=strategy,
                infer_table_structure=infer_table_structure,
                languages=langs,
            )
            return {"ok": True, "elements": _parts_to_elements(list(parts or []))}
        from langchain_community.document_loaders import UnstructuredPDFLoader

        docs = UnstructuredPDFLoader(
            str(path),
            mode="elements",
            strategy=strategy,
            infer_table_structure=infer_table_structure,
            languages=langs,
        ).load()
        elements: list[dict[str, Any]] = []
        for doc in docs:
            meta = dict(doc.metadata or {})
            html = meta.get("text_as_html") or meta.get("html") or ""
            if not isinstance(html, str):
                html = str(html or "")
            elements.append(
                {
                    "category": str(meta.get("category") or "Unknown"),
                    "page": meta.get("page_number") or meta.get("page") or 1,
                    "text": (doc.page_content or "").strip(),
                    "html": html.strip(),
                }
            )
        return {"ok": True, "elements": elements}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "elements": []}


def _parts_to_elements(parts: list[Any]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for el in parts:
        meta = getattr(el, "metadata", None)
        html = ""
        page = 1
        if meta is not None:
            html = getattr(meta, "text_as_html", None) or getattr(meta, "html", None) or ""
            page = getattr(meta, "page_number", None) or getattr(meta, "page", None) or 1
        if not isinstance(html, str):
            html = str(html or "")
        elements.append(
            {
                "category": str(getattr(el, "category", None) or type(el).__name__),
                "page": page,
                "text": str(getattr(el, "text", None) or "").strip(),
                "html": html.strip(),
            }
        )
    return elements
