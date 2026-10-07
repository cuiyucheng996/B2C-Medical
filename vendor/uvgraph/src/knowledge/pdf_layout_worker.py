"""子进程：Unstructured hi_res 锁定版面（标题/正文/表格/图片）。

图片直接 partition_image（YOLOX + OCR），PDF 才走 PDF Loader。
infer_table_structure 拆表；--layout-only 只标区域不拆格子。独立进程，避免把 API 打崩。
用法: python pdf_layout_worker.py --file a.jpg --out layout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_IMAGE_EXTS = {
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
    ".heic",
    ".heif",
    ".ico",
    ".avif",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _elements_from_parts(parts: list) -> list[dict[str, object]]:
    elements: list[dict[str, object]] = []
    for el in parts:
        meta = getattr(el, "metadata", None)
        html = ""
        page = 1
        if meta is not None:
            html = getattr(meta, "text_as_html", None) or getattr(meta, "html", None) or ""
            page = getattr(meta, "page_number", None) or getattr(meta, "page", None) or 1
        if not isinstance(html, str):
            html = str(html or "")
        cat = str(getattr(el, "category", None) or type(el).__name__)
        text = str(getattr(el, "text", None) or "").strip()
        elements.append(
            {
                "category": cat,
                "page": page,
                "text": text,
                "html": html.strip(),
            }
        )
    return elements


def run_layout(path: Path, *, layout_only: bool = False) -> dict:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import src.configs.config_loader  # noqa: F401 — Poppler / Tesseract / HF 缓存

    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTS:
        from unstructured.partition.image import partition_image

        parts = partition_image(
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=not layout_only,
            languages=["eng", "chi_sim"],
        )
        return {"ok": True, "elements": _elements_from_parts(list(parts or []))}

    from langchain_community.document_loaders import UnstructuredPDFLoader

    docs = UnstructuredPDFLoader(
        str(path),
        mode="elements",
        # hi_res：YOLOX 找标题/正文/表/图；ocr 识字；poppler 渲 PDF 页
        strategy="hi_res",
        infer_table_structure=not layout_only,
        languages=["eng", "chi_sim"],
    ).load()
    elements: list[dict[str, object]] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(description="版面分区 worker")
    parser.add_argument("--file", dest="path")
    parser.add_argument("--pdf", dest="pdf")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layout-only", dest="layout_only", action="store_true")
    args = parser.parse_args()
    target = Path(args.path or args.pdf or "")
    out = Path(args.out)
    try:
        if not target.is_file():
            raise FileNotFoundError(str(target))
        payload = run_layout(target, layout_only=bool(args.layout_only))
    except Exception as exc:
        payload = {"ok": False, "error": str(exc), "elements": []}
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if not payload.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()
