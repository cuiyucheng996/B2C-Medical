"""上传体检/化验 PDF 或图片：落盘 → 主项目抽字 → 抽指标行，供报告解读 Agent 使用。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["ReportIngest", "ingest_report_pdf", "format_report_user_text"]

_LAB_LINE = re.compile(
    r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9μ％%\-\(\)/]{1,24})"
    r"\s*([<>]?\d+(?:\.\d+)?)"
    r"(?:\s*([A-Za-zμ/%·升毫微]+))?"
    r"(?:\s*([↑↓高低偏]+))?"
)
_MAX_MD = 6000


@dataclass
class ReportIngest:
    filename: str
    saved_path: str
    parse_source: str
    markdown: str
    indicators: list[dict[str, str]] = field(default_factory=list)
    user_text: str = ""
    titles: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    image_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "saved_path": self.saved_path,
            "parse_source": self.parse_source,
            "indicator_count": len(self.indicators),
            "indicators": self.indicators[:40],
            "titles": self.titles[:20],
            "table_count": len(self.tables),
            "image_count": self.image_count,
            "markdown_preview": (self.markdown or "")[:800],
            "user_text": (self.user_text or "")[:2500],
        }


def ingest_report_pdf(*, user_id: str, filename: str, data: bytes) -> ReportIngest:
    """兼容入口：走统一文件前置，再收成报告结构。"""
    from server.builtin_server.report_read.gate import ingest_user_file

    bundle = ingest_user_file(user_id=user_id, filename=filename, data=data)
    return ReportIngest(
        filename=bundle.filename,
        saved_path=bundle.saved_path,
        parse_source=bundle.parse_source,
        markdown=bundle.markdown,
        indicators=bundle.indicators,
        user_text=bundle.user_text,
        titles=bundle.titles,
        tables=bundle.tables,
        image_count=bundle.image_count,
    )


_YOLOX_OK = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


def _ensure_raster_image(saved: Path, data: bytes) -> Path:
    """HEIC 等 YOLOX 不认的格式转成 PNG；jpg/png 原文件直送。"""
    suffix = saved.suffix.lower()
    if suffix in _YOLOX_OK:
        return saved
    from io import BytesIO

    from PIL import Image

    try:
        from pi_heif import register_heif_opener

        register_heif_opener()
    except Exception:
        pass
    image = Image.open(BytesIO(data))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    out = saved.with_suffix(".png")
    image.save(out, format="PNG")
    return out


def _pdf_extract_module():
    """按文件加载 pdf_extract，避免 import src.knowledge 时拉起向量库。"""
    import importlib.util

    ensure_uvgraph_root()
    from configs.paths_b2c import UVGRAPH_ROOT

    file_path = UVGRAPH_ROOT / "src" / "knowledge" / "pdf_extract.py"
    spec = importlib.util.spec_from_file_location("uvgraph_pdf_extract", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 pdf_extract")
    import sys

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_markdown(path: Path):
    mod = _pdf_extract_module()
    suffix = path.suffix.lower()
    if suffix in _YOLOX_OK or suffix in {".heic", ".heif", ".avif", ".ico"}:
        return mod.extract_image_markdown(path, try_mineru=False)
    return mod.extract_pdf_markdown(path, try_mineru=False)


def mineru_file_cache_key(path: Path) -> str:
    """同一 PDF 文件未变则复用 last_file 里的 MinerU 拆表结果。"""
    p = Path(path)
    st = p.stat()
    return f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}"


def _export_layout_images(path: Path) -> None:
    """PP-DocLayoutV2 的框图按页存成图片。"""
    from src.configs.cache import MINERU_OUTPUT_DIR, PROJECT_ROOT
    from src.configs.config_loader import get_yaml_config

    cfg = (get_yaml_config() or {}).get("mineru") or {}
    out_root = Path(str(cfg.get("output_dir") or MINERU_OUTPUT_DIR))
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    layouts = sorted(out_root.glob(f"{path.stem}/**/*_layout.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not layouts:
        return
    layout_pdf = layouts[0]
    dest = Path(__file__).resolve().parent / "Layout recognition image" / path.stem
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


def mineru_split_tables(path: Path, layout: str = "v2") -> tuple[list[dict], list[dict[str, str]], str]:
    """只拆表：MinerU → 结构化 JSON + 指标行。不再跑 YOLOX。"""
    mod = _pdf_extract_module()
    md = mod.mineru_markdown(path, layout=layout)
    try:
        _export_layout_images(path)
    except Exception:
        logger.warning("layout image export failed", exc_info=True)
    blobs = list(mod.extract_markdown_tables(md) or [])
    structured = [mod.table_to_json(item) for item in blobs]
    indicators = _extract_indicators(blobs, "")
    return structured, indicators, "mineru_table" if blobs else ""


def mineru_split_tables_cached(
    path: Path,
    row: dict[str, Any] | None,
    layout: str = "v2",
) -> tuple[list[dict], list[dict[str, str]], str]:
    """先进程内按 content_sha256 命中；再 last_file 同行；否则 MinerU 并写入内存。"""
    from algorithm.mineru_mem_cache import mineru_payload_from_mem, put_mineru_mem

    layout = "v3" if str(layout).lower() == "v3" else "v2"
    cached = row if isinstance(row, dict) else {}
    content_sha = str(cached.get("content_sha256") or "").strip()
    cache_key = f"{content_sha}:{layout}:fill" if content_sha else ""
    if cache_key:
        hit = mineru_payload_from_mem(cache_key)
        if hit is not None:
            return hit

    p = Path(path)
    key = mineru_file_cache_key(p)
    pdf_hint = str(cached.get("mineru_pdf_path") or cached.get("saved_path") or "")
    if (
        cached.get("mineru_cache_key") == key
        and pdf_hint
        and Path(pdf_hint).resolve() == p.resolve()
        and (cached.get("structured_tables") or cached.get("indicators"))
    ):
        tables = list(cached.get("structured_tables") or [])
        indicators = [dict(item) for item in (cached.get("indicators") or []) if isinstance(item, dict)]
        source = str(cached.get("parse_source") or ("mineru_table" if tables else ""))
        return tables, indicators, source

    tables, indicators, source = mineru_split_tables(p, layout=layout)
    if cache_key and (tables or indicators):
        put_mineru_mem(
            cache_key,
            structured_tables=tables,
            indicators=indicators,
            parse_source=source,
        )
    return tables, indicators, source


def _detect_layout(path: Path, *, layout_only: bool = True) -> dict[str, Any]:
    """YOLOX 版面分区：layout_only 时不拆格子，给预处理判断是不是报表。"""
    try:
        return dict(_pdf_extract_module().detect_layout(path, layout_only=layout_only) or {})
    except Exception as exc:
        logger.warning("layout detect failed: %s", exc)
        return {"ok": False, "table_count": 0, "error": str(exc)}


def _compose_limited_markdown(parsed: Any) -> str:
    tables = [str(item).strip() for item in (getattr(parsed, "tables", None) or []) if str(item).strip()]
    body = str(getattr(parsed, "body", "") or "").strip()
    titles = "\n".join(f"- {item}" for item in (getattr(parsed, "titles", None) or [])[:12])
    parts: list[str] = []
    if titles:
        parts.append("## 标题\n" + titles)
    if tables:
        parts.append("## 表格\n" + "\n\n".join(tables))
    head = "\n\n".join(parts)
    remain = _MAX_MD - len(head) - 20
    if body:
        if remain < 200:
            body = ""
        elif len(body) > remain:
            body = body[:remain] + "\n…（正文已截断）"
        parts.append("## 正文\n" + body)
    image_count = int(getattr(parsed, "image_count", 0) or 0)
    if image_count:
        parts.append(f"## 图片\n共 {image_count} 处图片区域（未把像素送进模型）")
    markdown = "\n\n".join(parts).strip() or str(getattr(parsed, "markdown", "") or "").strip()
    return markdown


def format_report_user_text(
    filename: str,
    markdown: str,
    indicators: list[dict[str, str]],
    tables: list[str] | None = None,
) -> str:
    lines = [f"【上传检查报告】{filename}", "请解读我上传的检查报告，化验单怎么看。"]
    if indicators:
        lines.append("从表格/正文抽出的指标（机器识别，可能有误）：")
        for item in indicators[:25]:
            arrow = item.get("flag") or ""
            unit = item.get("unit") or ""
            lines.append(f"- {item.get('name')} {item.get('value')} {unit} {arrow}".strip())
    table_blob = _join_whole_tables(tables or [], budget=4000)
    if table_blob:
        lines.append("表格结构（优先看这里）：")
        lines.append(table_blob)
    lines.append("分区摘录：")
    lines.append(markdown[:3500] if markdown else "（无正文）")
    lines.append("这不是诊断，有疑问要问医生。")
    return "\n".join(lines)


def _join_whole_tables(tables: list[str], *, budget: int) -> str:
    parts: list[str] = []
    used = 0
    for raw in tables:
        table = (raw or "").strip()
        if not table:
            continue
        if parts and used + len(table) + 2 > budget:
            break
        parts.append(table)
        used += len(table) + 2
        if used >= budget:
            break
    return "\n\n".join(parts)


def _extract_indicators(tables: list[str], body: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for table in tables or []:
        caption = _table_caption(table)
        for row in _table_rows(table):
            item = _row_to_indicator(row)
            if not item:
                continue
            key = f"{caption}:{item['name']}" if caption else item["name"]
            if key in seen:
                continue
            seen.add(key)
            if caption:
                item["group"] = caption
            found.append(item)
    for line in (body or "").splitlines():
        text = line.strip()
        if len(text) < 4:
            continue
        match = _LAB_LINE.search(text)
        if not match:
            continue
        name = match.group(1).strip()
        if name in seen or name.isdigit():
            continue
        seen.add(name)
        found.append(
            {
                "name": name,
                "value": match.group(2) or "",
                "unit": (match.group(3) or "").strip(),
                "flag": (match.group(4) or "").strip(),
            }
        )
    return found


def _table_caption(table: str) -> str:
    for line in (table or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            return text.lstrip("#").strip()
        if "<table" in text.lower() or text.startswith("|"):
            break
        if len(text) <= 24 and not text.startswith("<"):
            return text
        break
    return ""


def _table_rows(table: str) -> list[list[str]]:
    mod = _pdf_extract_module()
    rows = list(mod.table_rows(table or ""))
    return rows


def _html_table_rows(html: str) -> list[list[str]]:
    return _pdf_extract_module().html_table_rows(html or "")


def _row_to_indicator(cells: list[str]) -> dict[str, str] | None:
    if len(cells) < 2:
        return None
    name = cells[0].strip()
    if not name or name.isdigit() or len(name) > 24:
        return None
    if name in ("项目", "名称", "指标", "检查项", "Item", "比较内容", "组件"):
        return None
    value = ""
    unit = ""
    flag = ""
    for cell in cells[1:]:
        match = _LAB_LINE.search(f"{name} {cell}")
        if match and match.group(2):
            value = match.group(2)
            unit = (match.group(3) or "").strip()
            flag = (match.group(4) or "").strip()
            break
        if any(mark in cell for mark in ("↑", "↓", "高", "低", "偏")):
            flag = cell
        elif not value:
            value = cell
    if not value:
        return None
    return {"name": name, "value": value, "unit": unit, "flag": flag}
