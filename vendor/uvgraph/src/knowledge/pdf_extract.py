"""用户 PDF：先 Unstructured hi_res 分区，正文拼字符；表格交给 MinerU。"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "PdfExtractResult",
    "detect_layout",
    "extract_image_markdown",
    "extract_markdown_tables",
    "extract_pdf_markdown",
    "html_table_rows",
    "mineru_markdown",
    "table_rows",
    "table_to_json",
]

_HTML_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S.*$")
_CELL_TAG_RE = re.compile(r"<t([hd])(\s[^>]*)?>(.*?)</t\1>", re.I | re.S)
_SPAN_RE = re.compile(r"(rowspan|colspan)\s*=\s*['\"]?(\d+)", re.I)

_TITLE_CATS = frozenset({"Title", "Header", "Headline"})
_BODY_CATS = frozenset({"NarrativeText", "UncategorizedText", "ListItem", "Text", "Narrative"})
_TABLE_CATS = frozenset({"Table"})
_IMAGE_CATS = frozenset({"Image", "Figure", "Picture"})
_WORKER_TIMEOUT = 120
_MINERU_TIMEOUT = 90


@dataclass
class PdfExtractResult:
    markdown: str
    source: str
    page_hint: int = 0
    titles: list[str] = field(default_factory=list)
    body: str = ""
    tables: list[str] = field(default_factory=list)
    image_count: int = 0
    elements: list[dict] = field(default_factory=list)


def detect_layout(path: str | Path, *, layout_only: bool = True) -> dict:
    """YOLOX 版面分区。layout_only 时只标区域不拆格子，给「是不是报表」当依据。"""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在: {target}")
    elements = _run_layout_worker(target, layout_only=layout_only)
    counts: dict[str, int] = {}
    for item in elements:
        cat = str(item.get("category") or "")
        counts[cat] = counts.get(cat, 0) + 1
    return {
        "ok": bool(elements),
        "table_count": sum(int(counts.get(cat) or 0) for cat in _TABLE_CATS),
        "image_count": sum(int(counts.get(cat) or 0) for cat in _IMAGE_CATS),
        "title_count": sum(int(counts.get(cat) or 0) for cat in _TITLE_CATS),
        "categories": counts,
        "elements": elements,
    }


def extract_image_markdown(path: str | Path, *, try_mineru: bool = False) -> PdfExtractResult:
    """图片 YOLOX 分区。try_mineru 仅解释报表工具拆表时打开。"""
    image = Path(path)
    if not image.is_file():
        raise FileNotFoundError(f"图片不存在: {image}")
    layout = _run_layout_worker(image)
    titles, body, tables, image_count, pages = _split_regions(layout)
    source_bits = ["unstructured_image_hi_res"] if layout else []
    if try_mineru:
        mineru_tables = extract_markdown_tables(_try_mineru(image))
        if mineru_tables:
            tables = mineru_tables
            source_bits.append("mineru_table")
    if not source_bits:
        source_bits.append("empty")
    markdown = _assemble_markdown(image.name, titles, body, tables, image_count)
    return PdfExtractResult(
        markdown=markdown.strip(),
        source="+".join(source_bits),
        page_hint=pages,
        titles=titles,
        body=body.strip(),
        tables=tables,
        image_count=image_count,
        elements=list(layout or []),
    )


def extract_pdf_markdown(path: str | Path, *, try_mineru: bool = False) -> PdfExtractResult:
    pdf = Path(path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf}")

    layout = _run_layout_worker(pdf)
    titles, body, unstructured_tables, image_count, pages = _split_regions(layout)
    mineru_md = ""
    if try_mineru:
        mineru_md = _try_mineru(pdf)
    mineru_tables = extract_markdown_tables(mineru_md)
    tables = mineru_tables or unstructured_tables
    source_bits = []
    if layout:
        source_bits.append("unstructured_hi_res")
    if mineru_tables:
        source_bits.append("mineru_table")
    if not source_bits:
        source_bits.append("pypdf")
        pypdf_text, pages = _pypdf_text(pdf)
        body = pypdf_text
        layout = _elements_from_pages(pypdf_text, pages)
        if try_mineru and len(body.strip()) < 80:
            mineru_md = mineru_md or _try_mineru(pdf)
            tables = extract_markdown_tables(mineru_md)
            if tables:
                source_bits = ["pypdf", "mineru_table"]

    markdown = _assemble_markdown(pdf.name, titles, body, tables, image_count)
    if not markdown.replace("#", "").replace("图片", "").strip():
        raise ValueError("PDF 未能抽出文字，扫描件可稍后重试")
    return PdfExtractResult(
        markdown=markdown.strip(),
        source="+".join(source_bits),
        page_hint=pages,
        titles=titles,
        body=body.strip(),
        tables=tables,
        image_count=image_count,
        elements=list(layout or []),
    )


def _elements_from_pages(text: str, pages: int) -> list[dict]:
    chunks = (text or "").split("\n\n")
    elements: list[dict] = []
    total = max(int(pages or 1), 1)
    if not any((chunk or "").strip() for chunk in chunks):
        blob = (text or "").strip()
        return [{"category": "NarrativeText", "page": 1, "text": blob, "html": ""}] if blob else []
    for index, chunk in enumerate(chunks):
        if not (chunk or "").strip():
            continue
        elements.append(
            {
                "category": "NarrativeText",
                "page": min(index + 1, total),
                "text": chunk.strip(),
                "html": "",
            }
        )
    return elements


def _run_layout_worker(pdf: Path, *, layout_only: bool = False) -> list[dict]:
    worker = Path(__file__).resolve().parent / "pdf_layout_worker.py"
    out = pdf.with_suffix(pdf.suffix + ".layout.json")
    cmd = [sys.executable, str(worker), "--file", str(pdf), "--out", str(out)]
    if layout_only:
        cmd.append("--layout-only")
    try:
        completed = subprocess.run(cmd, timeout=_WORKER_TIMEOUT, capture_output=True, text=True)
    except Exception:
        logger.warning("pdf layout worker failed to start", exc_info=True)
        return []
    if completed.returncode != 0:
        logger.warning("pdf layout worker exit=%s stderr=%s", completed.returncode, (completed.stderr or "")[-400])
    if not out.is_file():
        return []
    try:
        payload = json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not payload.get("ok"):
        logger.warning("pdf layout worker ok=false error=%s", payload.get("error"))
        return []
    elements = payload.get("elements") or []
    return elements if isinstance(elements, list) else []


def _split_regions(elements: list[dict]) -> tuple[list[str], str, list[str], int, int]:
    titles: list[str] = []
    body_parts: list[str] = []
    tables: list[str] = []
    images = 0
    pages = 1
    last_heading = ""
    for item in elements:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category") or "")
        text = str(item.get("text") or "").strip()
        html = str(item.get("html") or "").strip()
        try:
            pages = max(pages, int(item.get("page") or 1))
        except (TypeError, ValueError):
            pass
        if cat in _IMAGE_CATS:
            images += 1
            continue
        if cat in _TITLE_CATS and text:
            last_heading = text
            if text not in titles:
                titles.append(text)
            continue
        if cat in _TABLE_CATS:
            table = html or text
            if table:
                tables.append(_with_caption(last_heading, table))
            continue
        if cat in _BODY_CATS or (text and cat not in _TITLE_CATS | _TABLE_CATS | _IMAGE_CATS):
            if text:
                body_parts.append(text)
    body = "\n".join(body_parts)
    return titles, body, tables, images, pages


def _assemble_markdown(
    name: str,
    titles: list[str],
    body: str,
    tables: list[str],
    image_count: int,
) -> str:
    lines = [f"# {name}", ""]
    if titles:
        lines.append("## 标题")
        lines.extend(f"- {item}" for item in titles[:20])
        lines.append("")
    if body.strip():
        lines.append("## 正文")
        lines.append(body.strip())
        lines.append("")
    if tables:
        lines.append("## 表格")
        for idx, table in enumerate(tables, start=1):
            lines.append(f"### 表{idx}")
            lines.append(table.strip())
            lines.append("")
    if image_count:
        lines.append("## 图片")
        lines.append(f"共 {image_count} 处图片区域（YOLOX 锁定，未把像素送进模型）")
        lines.append("")
    return "\n".join(lines)


def _pypdf_text(pdf: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    chunks = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(chunks), len(reader.pages)


def extract_markdown_tables(markdown: str) -> list[str]:
    """抽出整表：HTML <table> 或带分隔行的 GFM 表；绑上一节标题。行首单独的 | 不当表。"""
    if not (markdown or "").strip():
        return []
    units: list[tuple[str, str]] = []
    pos = 0
    spans = [(m.start(), m.end()) for m in _HTML_TABLE_RE.finditer(markdown)]

    def flush_text(start: int, end: int) -> None:
        if start < end:
            _split_pipe_tables(markdown[start:end], units)

    for a, b in spans:
        flush_text(pos, a)
        units.append(("table", markdown[a:b].strip()))
        pos = b
    flush_text(pos, len(markdown))

    tables: list[str] = []
    pending = ""
    for kind, unit in units:
        if kind == "table":
            tables.append(_with_caption(pending, unit))
            continue
        for line in unit.splitlines():
            if _HEADING_RE.match(line.strip()):
                pending = line.strip()
    return tables


def table_rows(table: str) -> list[list[str]]:
    raw = table or ""
    match = _HTML_TABLE_RE.search(raw)
    if match or "<tr" in raw.lower():
        rows = html_table_rows(match.group(0) if match else raw)
    else:
        rows = _pipe_table_rows(raw)
    return _fill_down_first_column(rows)


def html_table_rows(html: str) -> list[list[str]]:
    """展开 rowspan/colspan，合并单元格的项目名向下补齐。"""
    grid: list[list[str | None]] = []
    occupied: set[tuple[int, int]] = set()
    row_i = 0
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html or "", flags=re.I | re.S):
        while len(grid) <= row_i:
            grid.append([])
        col = 0
        for cell in _CELL_TAG_RE.finditer(tr):
            attrs = cell.group(2) or ""
            text = re.sub(r"<[^>]+>", "", cell.group(3) or "").strip()
            rowspan, colspan = _cell_spans(attrs)
            while (row_i, col) in occupied:
                col += 1
            for ri in range(rowspan):
                rr = row_i + ri
                while len(grid) <= rr:
                    grid.append([])
                for cj in range(colspan):
                    cc = col + cj
                    while len(grid[rr]) <= cc:
                        grid[rr].append(None)
                    occupied.add((rr, cc))
                    if grid[rr][cc] in (None, ""):
                        grid[rr][cc] = text
            col += colspan
        row_i += 1
    return [[(cell or "").strip() for cell in row] for row in grid if any((cell or "").strip() for cell in row)]


def _cell_spans(attrs: str) -> tuple[int, int]:
    rowspan = 1
    colspan = 1
    for name, value in _SPAN_RE.findall(attrs or ""):
        n = max(int(value), 1)
        if name.lower() == "rowspan":
            rowspan = n
        else:
            colspan = n
    return rowspan, colspan


def _pipe_table_rows(table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in (table or "").splitlines():
        text = line.strip()
        if not _PIPE_ROW_RE.match(text) or _is_pipe_sep(text):
            continue
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _is_pipe_sep(line: str) -> bool:
    stripped = line.strip()
    return "-" in stripped and bool(re.fullmatch(r"[\s:\-|]+", stripped))


def _split_pipe_tables(block: str, units: list[tuple[str, str]]) -> None:
    lines = block.splitlines(keepends=True)
    i = 0
    buf: list[str] = []

    def push_text() -> None:
        if buf:
            units.append(("text", "".join(buf).strip("\n")))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        if _PIPE_ROW_RE.match(line.rstrip("\n")):
            table_lines = [line]
            j = i + 1
            while j < len(lines) and _PIPE_ROW_RE.match(lines[j].rstrip("\n")):
                table_lines.append(lines[j])
                j += 1
            if len(table_lines) >= 2 and any(_is_pipe_sep(x.rstrip("\n")) for x in table_lines):
                push_text()
                units.append(("table", "".join(table_lines).strip("\n")))
                i = j
                continue
        buf.append(line)
        i += 1
    push_text()


def _with_caption(caption: str, table: str) -> str:
    cap = (caption or "").strip()
    body = (table or "").strip()
    if not cap or not body:
        return body
    if body.startswith(cap) or cap in body[:80]:
        return body
    return f"{cap}\n\n{body}"


def _fill_down_first_column(rows: list[list[str]]) -> list[list[str]]:
    prev = ""
    out: list[list[str]] = []
    for row in rows:
        cells = list(row)
        if not cells:
            continue
        if cells[0].strip():
            prev = cells[0].strip()
        elif prev:
            cells[0] = prev
        out.append(cells)
    return out


def table_to_json(table: str) -> dict:
    """一行表头 + 数据行 → {caption, headers, records}。"""
    caption = ""
    for line in (table or "").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            caption = text.lstrip("#").strip()
            break
        if "<table" in text.lower() or text.startswith("|"):
            break
        if len(text) <= 24 and not text.startswith("<"):
            caption = text
        break
    rows = table_rows(table)
    if not rows:
        return {"caption": caption, "headers": [], "records": []}
    headers = [cell or f"col_{i}" for i, cell in enumerate(rows[0])]
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        rec: dict[str, str] = {}
        for i, key in enumerate(headers):
            rec[key] = row[i] if i < len(row) else ""
        records.append(rec)
    return {"caption": caption, "headers": headers, "records": records}


def mineru_markdown(path: str | Path) -> str:
    """只跑 MinerU 拆表，不再做 YOLOX 分区。"""
    return _try_mineru(Path(path))


def _try_mineru(path: Path) -> str:
    """进程内 pipeline 拆表，用项目 .cache 本地权重，不启 mineru-api。"""
    try:
        from mineru.cli.common import do_parse

        from src.configs.cache import MINERU_OUTPUT_DIR, PROJECT_ROOT
        from src.configs.config_loader import get_yaml_config
    except Exception:
        logger.warning("mineru import skipped", exc_info=True)
        return ""
    cfg = (get_yaml_config() or {}).get("mineru") or {}
    out_root = Path(str(cfg.get("output_dir") or MINERU_OUTPUT_DIR))
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    backend = str(cfg.get("backend") or "pipeline")
    lang = str(cfg.get("lang") or "ch")
    try:
        do_parse(
            output_dir=str(out_root),
            pdf_file_names=[path.stem],
            pdf_bytes_list=[path.read_bytes()],
            p_lang_list=[lang],
            backend=backend,
            parse_method="auto",
            formula_enable=False,
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
        )
    except Exception:
        logger.warning("mineru table extract skipped or failed", exc_info=True)
        return ""
    matches = sorted(out_root.rglob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    stem = path.stem
    for item in matches:
        if stem in item.stem or stem in str(item.parent):
            text = item.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return text
    if matches:
        return matches[0].read_text(encoding="utf-8", errors="ignore").strip()
    return ""
