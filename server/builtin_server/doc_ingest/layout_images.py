"""MinerU 版面框图导出到 doc_ingest/Layout recognition image/<PDF 文件名>/。"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_LAYOUT_ROOT = Path(__file__).resolve().parent / "Layout recognition image"


def export_mineru_layout_images(pdf_filename: str) -> Path | None:
    """从 .cache/mineru/output 找最新 layout PDF，按页导出 page_XX.png。"""
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs.cache import MINERU_OUTPUT_DIR, PROJECT_ROOT
    from src.configs.config_loader import get_yaml_config

    stem = Path(pdf_filename or "upload").stem
    cfg = (get_yaml_config() or {}).get("mineru") or {}
    out_root = Path(str(cfg.get("output_dir") or MINERU_OUTPUT_DIR))
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    layouts = sorted(
        (p for p in out_root.rglob("*_layout.pdf") if stem in p.as_posix()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not layouts:
        logger.info("mineru layout pdf not found for %s under %s", stem, out_root)
        return None
    layout_pdf = layouts[0]
    dest = _LAYOUT_ROOT / stem
    dest.mkdir(parents=True, exist_ok=True)
    target_pdf = dest / layout_pdf.name
    if layout_pdf.resolve() != target_pdf.resolve():
        target_pdf.write_bytes(layout_pdf.read_bytes())
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(layout_pdf))
    try:
        for index in range(len(doc)):
            page = doc[index]
            bitmap = page.render(scale=2)
            bitmap.to_pil().save(dest / f"page_{index + 1:02d}.png")
    finally:
        doc.close()
    return dest
