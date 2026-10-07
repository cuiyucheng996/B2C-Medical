"""上传文件前置（预处理）：落盘分型 → 文档加载 → 再判像不像化验单。

文档加载（txt/md/docx/html/web，以及文字型 PDF 快路径）统一走
algorithm.document_load；图片/扫描 PDF 仍走 report_ingest（YOLOX）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from configs.paths_b2c import B2C_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "FileIngest",
    "classify_file_family",
    "classify_file_role",
    "ingest_user_file",
    "intent_for_file_role",
    "parse_user_file",
    "score_upload_role",
]

_TITLE_CATS = frozenset({"Title", "Header", "Headline"})
_BODY_CATS = frozenset({"NarrativeText", "UncategorizedText", "ListItem", "Text", "Narrative"})
_TABLE_CATS = frozenset({"Table"})
_IMAGE_CATS = frozenset({"Image", "Figure", "Picture"})

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
_DOC_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".ppt",
    ".pptx",
    ".rtf",
    ".odt",
}
_OFFICE_ZIP = {".docx", ".xlsx", ".pptx"}
_MAX_BYTES = 15 * 1024 * 1024
_LAB_HINTS = (
    "检验",
    "报告单",
    "化验",
    "参考范围",
    "参考值",
    "mmol",
    "u/l",
    "白细胞",
    "红细胞",
    "胆固醇",
    "转氨酶",
    "空腹血糖",
    "血红蛋白",
    "血常规",
    "尿常规",
    "生化",
    "甲状腺",
    "甲功",
    "体检",
)

FileFamily = Literal["image", "document"]
FileRole = Literal["lab_report", "general_image", "general_document", "layout_picture"]


@dataclass
class FileIngest:
    filename: str
    saved_path: str
    family: FileFamily
    role: FileRole
    reasons: list[str] = field(default_factory=list)
    parse_source: str = ""
    markdown: str = ""
    indicators: list[dict[str, str]] = field(default_factory=list)
    user_text: str = ""
    titles: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    body: str = ""
    image_count: int = 0
    table_count: int = 0
    user_id: str = ""
    chunks: list[dict[str, Any]] = field(default_factory=list)
    mineru_pdf_path: str = ""
    content_sha256: str = ""
    mineru_cache_key: str = ""
    structured_tables: list[dict[str, Any]] = field(default_factory=list)
    score_card: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "saved_path": self.saved_path,
            "family": self.family,
            "role": self.role,
            "reasons": list(self.reasons),
            "parse_source": self.parse_source,
            "indicator_count": len(self.indicators),
            "indicators": self.indicators[:40],
            "titles": self.titles[:20],
            "tables": [str(item)[:800] for item in self.tables[:8]],
            "body": (self.body or "")[:1500],
            "table_count": self.table_count,
            "image_count": self.image_count,
            "user_id": self.user_id,
            "chunk_count": len(self.chunks),
            "chunks": [_preview_chunk(item) for item in self.chunks[:80]],
            "mineru_pdf_path": self.mineru_pdf_path,
            "content_sha256": self.content_sha256,
            "mineru_cache_key": self.mineru_cache_key,
            "structured_tables": [dict(item) for item in self.structured_tables[:8]],
            "score_card": dict(self.score_card or {}),
            "markdown_preview": (self.markdown or "")[:800],
            "user_text": (self.user_text or "")[:2500],
        }

    def to_report_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        row.pop("family", None)
        row.pop("role", None)
        row.pop("reasons", None)
        return row


def _hydrate_mineru_from_memory(ingest: FileIngest) -> FileIngest:
    """同内容再次上传：只读进程内 MinerU 缓存，不跑 GPU。"""
    sha = (ingest.content_sha256 or "").strip()
    if not sha:
        return ingest
    from algorithm.mineru_mem_cache import mineru_payload_from_mem

    hit = mineru_payload_from_mem(sha)
    if hit is None:
        return ingest
    structured, indicators, source = hit
    ingest.structured_tables = list(structured or [])
    if indicators:
        ingest.indicators = list(indicators)
    if source:
        ingest.parse_source = source
    if ingest.structured_tables and not ingest.table_count:
        ingest.table_count = len(ingest.structured_tables)
    return ingest


def _preview_chunk(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    text = str(row.get("text") or "")
    if len(text) > 400:
        row["text"] = text[:400]
    html = str(row.get("html") or "")
    if len(html) > 400:
        row["html"] = html[:400]
    return row


def classify_file_family(filename: str, data: bytes) -> FileFamily:
    """后缀 + 文件头：常见图片，或 PDF/Office/文本。"""
    name = Path(filename or "").name
    suffix = Path(name).suffix.lower()
    magic = _sniff(data)
    if suffix == ".pdf" or magic == "pdf":
        if magic and magic != "pdf":
            raise ValueError("扩展名是 PDF，但文件内容不是 PDF")
        return "document"
    if magic == "image" or suffix in _IMAGE_EXTS:
        if suffix in _DOC_EXTS and magic == "image":
            raise ValueError("扩展名是文档，但文件内容是图片")
        return "image"
    if suffix in _OFFICE_ZIP:
        if magic and magic != "zip":
            raise ValueError("Word/Excel/PPT 内容损坏或不是该格式")
        return "document"
    if suffix in {".doc", ".xls", ".ppt"}:
        if magic and magic not in {"ole", "zip"}:
            raise ValueError("老版 Office 文件内容对不上")
        return "document"
    if suffix in _DOC_EXTS:
        return "document"
    raise ValueError(
        "目前支持常见图片（jpg/png/webp/gif/heic 等）和文档（pdf/word/excel/ppt/txt/md/html/csv）"
    )


def score_upload_role(
    *,
    family: FileFamily,
    table_count: int = 0,
    indicator_count: int = 0,
    text: str = "",
    layout_pending: bool = False,
    lab_threshold: int = 2,
) -> dict[str, Any]:
    """上传打分：表 +2、指标≥3 +2、化验词 +1、版面未拆开 +1；总分 ≥ 阈值（默认 2）才标 lab_report。"""
    points: list[dict[str, Any]] = []
    reasons: list[str] = []
    score = 0
    if table_count >= 1:
        score += 2
        detail = f"表格{table_count}处"
        points.append({"name": "表格区域", "delta": 2, "hit": True, "detail": detail})
        reasons.append(detail)
    else:
        points.append({"name": "表格区域", "delta": 0, "hit": False, "detail": "无"})
    if indicator_count >= 3:
        score += 2
        detail = f"指标{indicator_count}条"
        points.append({"name": "抽出指标", "delta": 2, "hit": True, "detail": detail})
        reasons.append(detail)
    else:
        points.append({"name": "抽出指标", "delta": 0, "hit": False, "detail": f"{indicator_count}条"})
    blob = (text or "").replace(" ", "").lower()
    hits = [hint for hint in _LAB_HINTS if hint.replace(" ", "").lower() in blob]
    if hits:
        score += 1
        detail = ",".join(hits[:4])
        points.append({"name": "文件名/正文化验词", "delta": 1, "hit": True, "detail": detail})
        reasons.append("命中化验词:" + detail)
    else:
        points.append({"name": "文件名/正文化验词", "delta": 0, "hit": False, "detail": "未命中"})
    if layout_pending:
        score += 1
        points.append({"name": "版面待抽表", "delta": 1, "hit": True, "detail": "YOLOX 未拆开"})
        reasons.append("待抽表原件")
    else:
        points.append({"name": "版面待抽表", "delta": 0, "hit": False, "detail": "已拆开或未走版面"})
    thresh = max(2, int(lab_threshold if lab_threshold is not None else 2))
    lab = score >= thresh
    if lab:
        role: FileRole = "lab_report"
    elif family == "image":
        role = "general_image"
    else:
        role = "general_document"
    return {
        "score": score,
        "threshold": thresh,
        "lab_report": lab,
        "role": role,
        "points": points,
        "reasons": reasons,
    }


def classify_file_role(
    *,
    family: FileFamily,
    table_count: int = 0,
    indicator_count: int = 0,
    text: str = "",
    layout_pending: bool = False,
) -> tuple[FileRole, list[str]]:
    """内容判定：化验单 vs 普通图 vs 普通 PDF。"""
    packed = score_upload_role(
        family=family,
        table_count=table_count,
        indicator_count=indicator_count,
        text=text,
        layout_pending=layout_pending,
    )
    role = packed["role"]
    reasons = list(packed["reasons"] or [])
    if role == "lab_report":
        return "lab_report", reasons or ["结构化像化验单"]
    if family == "image":
        return "general_image", reasons or ["图片且无化验结构"]
    return "general_document", reasons or ["文档且无化验结构"]


def ingest_user_file(
    *,
    user_id: str,
    filename: str,
    data: bytes,
    layout_only: bool = True,
    lab_threshold: int = 2,
) -> FileIngest:
    uid = (user_id or "").strip() or "unknown"
    name = Path(filename or "upload.bin").name
    if not data:
        raise ValueError("空文件")
    if len(data) > _MAX_BYTES:
        raise ValueError("文件超过 15MB")
    family = classify_file_family(name, data)

    dest_dir = B2C_ROOT / "data" / "user_files" / uid
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid4().hex}_{name}"
    dest.write_bytes(data)
    from algorithm.mineru_mem_cache import file_content_sha256

    content_sha256 = file_content_sha256(data)

    suffix = Path(name).suffix.lower()
    scan = family == "image" or suffix == ".pdf"
    layout_pending = False
    table_count = 0
    titles: list[str] = []
    body = ""
    tables: list[str] = []
    chunks: list[dict[str, Any]] = []
    markdown = ""
    image_count = 0
    mineru_pdf_path = ""
    layout_picture = False
    saved_path = str(dest)
    probe_path = dest
    if scan:
        from algorithm.document_load import stamp_chunks
        from algorithm.report_ingest import _compose_limited_markdown, _detect_layout, _ensure_raster_image

        if family == "image":
            try:
                probe_path = _ensure_raster_image(dest, data)
            except Exception as exc:
                raise ValueError(f"图片打不开：{exc}") from exc
        layout = _detect_layout(probe_path, layout_only=layout_only)
        elements = list(layout.get("elements") or []) if isinstance(layout, dict) else []
        if layout.get("ok") and elements:
            titles, body, tables, image_count, table_count = _split_layout_elements(elements)
            chunks = stamp_chunks(elements, user_id=uid, saved_path=str(dest))
            from types import SimpleNamespace

            markdown = _compose_limited_markdown(
                SimpleNamespace(titles=titles, body=body, tables=tables, image_count=image_count, markdown="")
            )
        elif layout.get("ok"):
            table_count = int(layout.get("table_count") or 0)
        else:
            layout_pending = True
    else:
        from algorithm.report_ingest import _compose_limited_markdown

        office = _extract_office(dest, user_id=uid)
        titles = list(getattr(office, "titles", None) or [])
        body = str(getattr(office, "body", "") or "")
        tables = list(getattr(office, "tables", None) or [])
        table_count = len(tables)
        markdown = _compose_limited_markdown(office)
        from algorithm.document_load import chunks_from_documents, stamp_chunks

        docs = list(getattr(office, "documents", None) or [])
        if docs:
            chunks = chunks_from_documents(docs, user_id=uid, saved_path=str(dest))
        else:
            chunks = stamp_chunks(
                [{"category": "Table", "page": 1, "text": "", "html": item} for item in tables]
                + ([{"category": "NarrativeText", "page": 1, "text": body, "html": ""}] if body else []),
                user_id=uid,
                saved_path=str(dest),
            )
    card = score_upload_role(
        family=family,
        table_count=table_count,
        indicator_count=0,
        text="\n".join([name, *titles, body]),
        layout_pending=layout_pending,
        lab_threshold=lab_threshold,
    )
    role = card["role"]
    reasons = list(card["reasons"] or [])
    if role == "lab_report" and not reasons:
        reasons = ["结构化像化验单"]
    picture_only = bool(scan and image_count >= 1 and table_count < 1)
    if role == "lab_report":
        mineru_pdf_path = _ensure_report_pdf(dest=dest, probe=probe_path)
        layout_picture = False
    elif picture_only:
        _discard_user_file(dest)
        if probe_path != dest:
            _discard_user_file(probe_path)
        saved_path = ""
        chunks = []
        titles = []
        tables = []
        body = ""
        markdown = ""
        mineru_pdf_path = ""
        layout_picture = True
        role = "layout_picture"
        reasons = ["版面为图片区域，不保存、不解读"]
        card = dict(card)
        card["role"] = role
        card["overridden"] = "layout_picture"
    bundle = FileIngest(
        filename=name,
        saved_path=saved_path,
        family=family,
        role=role,
        reasons=reasons,
        parse_source="preview",
        markdown=markdown,
        user_text=_format_pending_user_text(
            name, family, role, table_count, titles, tables, body, chunks, mineru_pdf_path
        ),
        titles=titles,
        tables=tables[:8],
        body=body,
        image_count=image_count,
        table_count=table_count,
        user_id=uid,
        chunks=chunks,
        mineru_pdf_path=mineru_pdf_path,
        content_sha256=content_sha256,
        score_card=card,
    )
    return _hydrate_mineru_from_memory(bundle)


def parse_user_file(*, saved_path: str, filename: str = "", family: str = "", user_id: str = "") -> FileIngest:
    """Agent 工具侧抽表：YOLOX / 统一文档加载，与旧预处理同一套解析。"""
    dest = Path(saved_path)
    if not dest.is_file():
        raise ValueError("原件不存在或已过期")
    name = Path(filename or dest.name).name
    data = dest.read_bytes()
    from algorithm.mineru_mem_cache import file_content_sha256

    content_sha = file_content_sha256(data)
    fam: FileFamily = "image" if family == "image" else "document"
    if not family:
        fam = classify_file_family(name, data)
    uid = (user_id or dest.parent.name or "").strip()

    from algorithm.report_ingest import (
        _compose_limited_markdown,
        _ensure_raster_image,
        _extract_indicators,
        _extract_markdown,
        _pdf_extract_module,
        format_report_user_text,
    )

    parse_path = dest
    suffix = dest.suffix.lower()
    if fam == "image":
        parse_path = _ensure_raster_image(dest, data)
        parsed = _extract_markdown(parse_path)
    elif suffix == ".pdf":
        parsed = _extract_markdown(parse_path)
    else:
        parsed = _extract_office(parse_path, user_id=uid)
    tables = list(getattr(parsed, "tables", None) or [])
    body = str(getattr(parsed, "body", "") or "")
    titles = list(getattr(parsed, "titles", None) or [])
    if not tables:
        try:
            extract_tables = getattr(_pdf_extract_module(), "extract_markdown_tables", None)
            md = str(getattr(parsed, "markdown", "") or "")
            if extract_tables and md:
                tables = list(extract_tables(md) or [])
        except Exception:
            pass
    markdown = _compose_limited_markdown(parsed)
    indicators = _extract_indicators(tables, body)
    role, reasons = classify_file_role(
        family=fam,
        table_count=len(tables),
        indicator_count=len(indicators),
        text="\n".join([*titles, body, markdown, name]),
    )
    if role == "lab_report":
        user_text = format_report_user_text(name, markdown, indicators, tables)
    elif fam == "image":
        user_text = _format_image_user_text(name, markdown, body)
    else:
        user_text = _format_document_user_text(name, markdown, body)
    from algorithm.document_load import chunks_from_documents, stamp_chunks

    docs = list(getattr(parsed, "documents", None) or [])
    if docs:
        chunks = chunks_from_documents(docs, user_id=uid, saved_path=str(dest))
    else:
        chunks = stamp_chunks(
            list(getattr(parsed, "elements", None) or []),
            user_id=uid,
            saved_path=str(dest),
        )
    if not chunks and (body or titles or tables):
        chunks = stamp_chunks(
            [
                {
                    "category": "NarrativeText",
                    "page": 1,
                    "text": body or "\n".join(titles),
                    "html": (tables[0] if tables else ""),
                }
            ],
            user_id=uid,
            saved_path=str(dest),
        )
    mineru_pdf = ""
    if role == "lab_report":
        mineru_pdf = _ensure_report_pdf(dest=dest, probe=parse_path)
    bundle = FileIngest(
        filename=name,
        saved_path=str(dest),
        family=fam,
        role=role,
        reasons=reasons,
        parse_source=str(getattr(parsed, "source", "") or ""),
        markdown=markdown,
        indicators=indicators,
        user_text=user_text,
        titles=titles,
        tables=tables[:8],
        image_count=int(getattr(parsed, "image_count", 0) or 0),
        table_count=len(tables),
        user_id=uid,
        chunks=chunks,
        mineru_pdf_path=mineru_pdf,
        content_sha256=content_sha,
    )
    return _hydrate_mineru_from_memory(bundle)


def intent_for_file_role(role: str) -> dict[str, str] | None:
    """前置分流对应的细意图，供 Agent 覆盖 LLM 分类。"""
    mapping = {
        "lab_report": {
            "intent": "上传检查报告并解读",
            "major": "报告指标解读",
            "handler": "report_upload",
        },
        "general_image": {
            "intent": "解读用户上传的日常图片",
            "major": "近况陈述",
            "handler": "file_image",
        },
        "general_document": {
            "intent": "解读用户上传的普通文档",
            "major": "近况陈述",
            "handler": "file_document",
        },
        "layout_picture": {
            "intent": "寒暄打招呼",
            "major": "闲聊",
            "handler": "chat_hello",
        },
    }
    return mapping.get(role)


def _probe_office_tables(path: Path) -> int:
    """Office/文本只数表，不组正文：报告大意图必须有表。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            from docx import Document

            return len(Document(str(path)).tables)
        if suffix == ".xlsx":
            from openpyxl import load_workbook

            book = load_workbook(str(path), read_only=True, data_only=True)
            count = 0
            try:
                for name in book.sheetnames[:6]:
                    rows = 0
                    for row in book[name].iter_rows(values_only=True):
                        if any(cell is not None and str(cell).strip() for cell in row):
                            rows += 1
                        if rows >= 2:
                            break
                    if rows >= 2:
                        count += 1
            finally:
                book.close()
            return count
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
                import csv

                rows = sum(1 for _ in csv.reader(handle) if _)
            return 1 if rows >= 2 else 0
    except Exception as exc:
        logger.warning("office table probe failed %s: %s", path.name, exc)
    return 0


def _sniff(data: bytes) -> str:
    head = data[:16] if data else b""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8") or head.startswith(b"GIF8"):
        return "image"
    if head.startswith(b"BM") or head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
        return "image"
    if head.startswith(b"\x00\x00\x01\x00") or head.startswith(b"\x00\x00\x02\x00"):
        return "image"
    if head.startswith(b"RIFF") and b"WEBP" in data[:16]:
        return "image"
    if len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in {
        b"heic",
        b"heif",
        b"mif1",
        b"msf1",
        b"avif",
    }:
        return "image"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "ole"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    return ""


def _split_layout_elements(elements: list[dict[str, Any]]) -> tuple[list[str], str, list[str], int, int]:
    titles: list[str] = []
    body_parts: list[str] = []
    tables: list[str] = []
    images = 0
    table_boxes = 0
    for item in elements:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("category") or "")
        text = str(item.get("text") or "").strip()
        html = str(item.get("html") or "").strip()
        if cat in _IMAGE_CATS:
            images += 1
            continue
        if cat in _TITLE_CATS and text:
            if text not in titles:
                titles.append(text)
            continue
        if cat in _TABLE_CATS:
            table_boxes += 1
            blob = html or text
            if blob:
                tables.append(blob)
            continue
        if cat in _BODY_CATS or (text and cat not in _TITLE_CATS | _TABLE_CATS | _IMAGE_CATS):
            if text:
                body_parts.append(text)
    return titles, "\n".join(body_parts), tables, images, table_boxes or len(tables)


def _ensure_report_pdf(*, dest: Path, probe: Path | None = None) -> str:
    """报告解读用：已是 PDF 用原件；图片能转就转一份。转不了返回空。"""
    dest = Path(dest)
    src = Path(probe) if probe is not None and Path(probe).is_file() else dest
    if dest.suffix.lower() == ".pdf" and dest.is_file():
        return str(dest)
    if src.suffix.lower() == ".pdf" and src.is_file():
        return str(src)
    if src.suffix.lower() not in _IMAGE_EXTS:
        return ""
    try:
        from PIL import Image

        image = Image.open(src)
        if image.mode != "RGB":
            image = image.convert("RGB")
        out = dest.with_name(f"{dest.stem}_mineru.pdf")
        image.save(out, format="PDF")
        if out.is_file() and out.stat().st_size > 0:
            return str(out)
    except Exception:
        logger.warning("report file cannot convert to pdf", exc_info=True)
    return ""


def _pdf_for_mineru_if_table(*, dest: Path, probe: Path, table_count: int) -> str:
    """版面已标出表时转 PDF；无表则空。"""
    if table_count < 1:
        return ""
    return _ensure_report_pdf(dest=dest, probe=probe)


def _discard_user_file(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        logger.warning("discard layout picture failed path=%s", path)


def _format_pending_user_text(
    filename: str,
    family: FileFamily,
    role: FileRole,
    table_count: int = 0,
    titles: list[str] | None = None,
    tables: list[str] | None = None,
    body: str = "",
    chunks: list[dict[str, Any]] | None = None,
    mineru_pdf_path: str = "",
) -> str:
    chunk_n = len(chunks or [])
    sample = chunks[0] if chunks else {}
    stamp = ""
    if sample:
        stamp = (
            f"分区 {chunk_n} 块，user_id={sample.get('user_id') or ''}，"
            f"path={sample.get('saved_path') or ''}，含页码/元素类型。"
        )
    if role == "lab_report":
        lines = [
            f"【上传检查报告】{filename}",
            stamp or f"版面已标出表格 {table_count} 处。",
        ]
        if mineru_pdf_path:
            lines.append(f"已转 PDF 供拆表：{mineru_pdf_path}")
            lines.append("explain_uploaded_report 用 MinerU 读该 PDF，不要再做版面检测。")
        else:
            lines.append("请基于下列标题/正文/表解读，不要再解析原件。")
        if titles:
            lines.append("标题：" + "；".join(titles[:8]))
        if tables:
            lines.append("表格摘录：")
            lines.extend(str(item)[:600] for item in tables[:3])
        elif body.strip():
            lines.append("正文摘录：")
            lines.append(body.strip()[:1200])
        return "\n".join(lines)
    if role == "layout_picture":
        return "\n".join(
            [
                f"【上传图片】{filename}",
                "版面为图片区域，未保存原件、未做解读。按闲聊回应即可。",
            ]
        )
    if family == "image":
        return "\n".join(
            [
                f"【上传图片】{filename}",
                stamp or "版面已分区。",
                (body or "")[:1200] or "请根据分区说明画面。",
            ]
        )
    return "\n".join(
        [
            f"【上传文档】{filename}",
            stamp or "文档已切开。",
            (body or "")[:1200] or "请根据分区做摘要。",
        ]
    )


def _format_image_user_text(filename: str, markdown: str, body: str) -> str:
    ocr = (body or markdown or "").strip() or "（几乎没有识别出文字）"
    return "\n".join(
        [
            f"【上传图片】{filename}",
            "请根据识别出的画面/文字，说说这张图里有什么。不是化验单。",
            "识别摘录：",
            ocr[:3500],
        ]
    )


def _format_document_user_text(filename: str, markdown: str, body: str) -> str:
    ocr = (markdown or body or "").strip() or "（几乎没有识别出文字）"
    return "\n".join(
        [
            f"【上传文档】{filename}",
            "请根据文档内容做摘要说明。不是化验单。",
            "识别摘录：",
            ocr[:3500],
        ]
    )


def _extract_office(path: Path, *, user_id: str = "") -> Any:
    """Word / Excel / 文本等：抽出正文和表，形状对齐 pdf_extract 的结果。"""
    from types import SimpleNamespace

    suffix = path.suffix.lower()
    documents: list = []
    try:
        if suffix in {".txt", ".md", ".markdown", ".docx", ".html", ".htm"}:
            from algorithm.document_load import load_user_source

            loaded = load_user_source(path, user_id=user_id)
            body, titles, tables = loaded.body, loaded.titles, loaded.tables
            source = loaded.loader
            documents = list(loaded.documents)
            # docx 表格 unstructured 常漏，python-docx 补抽
            if suffix == ".docx" and not tables:
                _, _, tables = _read_docx(path)
        elif suffix == ".xlsx":
            body, titles, tables = _read_xlsx(path)
            source = "xlsx"
        elif suffix == ".csv":
            body, titles, tables = _read_csv(path)
            source = "csv"
        else:
            body, titles, tables = _read_unstructured(path)
            source = "unstructured"
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"文档解析失败：{exc}") from exc
    return SimpleNamespace(
        source=source,
        body=body,
        titles=titles,
        tables=tables,
        image_count=0,
        markdown="",
        documents=documents,
    )


def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _html_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    parts = ["<table><tbody>"]
    for row in rows[:80]:
        parts.append("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _read_docx(path: Path) -> tuple[str, list[str], list[str]]:
    from docx import Document

    doc = Document(str(path))
    titles = [p.text.strip() for p in doc.paragraphs[:8] if p.text.strip()][:8]
    body = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    tables: list[str] = []
    for table in doc.tables[:8]:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        html = _html_table(rows)
        if html:
            tables.append(html)
    return body, titles, tables


def _read_xlsx(path: Path) -> tuple[str, list[str], list[str]]:
    from openpyxl import load_workbook

    book = load_workbook(str(path), read_only=True, data_only=True)
    titles = list(book.sheetnames)[:8]
    tables: list[str] = []
    texts: list[str] = []
    for name in book.sheetnames[:6]:
        sheet = book[name]
        rows: list[list[str]] = []
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i >= 80:
                break
            cells = ["" if cell is None else str(cell).strip() for cell in row]
            if any(cells):
                rows.append(cells)
        html = _html_table(rows)
        if html:
            tables.append(html)
        texts.append(name)
    book.close()
    return "\n".join(texts), titles, tables


def _read_csv(path: Path) -> tuple[str, list[str], list[str]]:
    import csv

    rows: list[list[str]] = []
    with path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
        for i, row in enumerate(csv.reader(handle)):
            if i >= 80:
                break
            cells = [str(cell).strip() for cell in row]
            if any(cells):
                rows.append(cells)
    html = _html_table(rows)
    return "", [], [html] if html else []


def _read_unstructured(path: Path) -> tuple[str, list[str], list[str]]:
    from unstructured.partition.auto import partition

    parts = partition(filename=str(path))
    texts = [str(getattr(item, "text", "") or "").strip() for item in parts]
    body = "\n".join(item for item in texts if item)
    titles = [item for item in texts if item][:8]
    return body, titles, []

