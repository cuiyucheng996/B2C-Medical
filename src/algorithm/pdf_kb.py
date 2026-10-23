"""本地文档库：按 RAGFlow naive 思路把 PDF 切成块，RAGFlow 服务未起时也能灌库检索。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from configs.paths_b2c import B2C_ROOT, ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "append_local_text_chunks",
    "describe_pdf_preprocess",
    "ingest_local_pdf",
    "list_local_documents",
    "load_local_chunks",
    "retrieve_local",
    "retrieve_local_hits",
]

_KB_DIR = B2C_ROOT / "model" / "local_vector_db"
_CHUNKS = _KB_DIR / "chunks.jsonl"
_MANIFEST = _KB_DIR / "manifest.json"


def _chunk_chars() -> int:
    try:
        from algorithm.chunk_config import get_chunk_config

        return int(get_chunk_config().get("local_naive_chars") or 500)
    except Exception:
        return 500


def _chunk_overlap() -> int:
    try:
        from algorithm.chunk_config import get_chunk_config

        return int(get_chunk_config().get("local_overlap") or 60)
    except Exception:
        return 60


def describe_pdf_preprocess(parsed: Any, *, with_tsr: bool = False) -> dict[str, Any]:
    """入库切块前：转图抽字 → 版面 → 去页眉页脚 → 阅读序 content_blocks。"""
    pages = list(getattr(parsed, "pages", None) or [])
    layouts = list(getattr(parsed, "layouts", None) or [])
    blocks = list(getattr(parsed, "content_blocks", None) or [])
    char_n = sum(len(getattr(page, "chars", None) or []) for page in pages)
    page_chars: dict[int, dict[str, int]] = {}
    for page in pages:
        num = int(getattr(page, "page_number", 0) or 0)
        native = 0
        ocr = 0
        for c in page.chars or []:
            if not isinstance(c, dict):
                continue
            if str(c.get("source") or "").startswith("ocr"):
                ocr += 1
            else:
                native += 1
        page_chars[num] = {"native": native, "ocr": ocr}
    native_n = sum(item["native"] for item in page_chars.values())
    ocr_n = sum(item["ocr"] for item in page_chars.values())
    ocr_pages = sum(1 for item in page_chars.values() if item["ocr"])
    labels: dict[str, int] = {}
    line_n = 0
    backends: dict[str, int] = {}
    page_rows: list[dict[str, Any]] = []
    for layout in layouts:
        backend = str(getattr(layout, "backend", "") or "-")
        backends[backend] = backends.get(backend, 0) + 1
        lines = len(getattr(layout, "line_boxes", None) or [])
        line_n += lines
        regions = list(getattr(layout, "regions", None) or [])
        for region in regions:
            lab = str(getattr(region, "label", None) or getattr(region, "layout_type", None) or "other")
            labels[lab] = labels.get(lab, 0) + 1
        page_no = int(getattr(layout, "page_number", 0) or 0)
        page = next((p for p in pages if int(getattr(p, "page_number", 0) or 0) == page_no), None)
        page_rows.append(
            {
                "page": getattr(layout, "page_number", None),
                "chars": len(getattr(page, "chars", None) or []) if page is not None else 0,
                "backend": backend,
                "regions": len(regions),
                "lines": lines,
                "native": page_chars.get(page_no, {}).get("native", 0),
                "ocr": page_chars.get(page_no, {}).get("ocr", 0),
            }
        )
    kinds: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for block in blocks:
        kind = str(block.get("type") or "other")
        kinds[kind] = kinds.get(kind, 0) + 1
        if len(samples) < 12:
            text = str(block.get("text") or block.get("caption") or "")[:120]
            samples.append({"type": kind, "page": block.get("page"), "text": text})
    label_txt = "、".join(f"{k}:{v}" for k, v in sorted(labels.items())) or "无"
    kind_txt = "、".join(f"{k}:{v}" for k, v in sorted(kinds.items())) or "无"
    backend_txt = "、".join(f"{k}×{v}" for k, v in sorted(backends.items())) or "-"
    zoom = float(getattr(parsed, "zoom", 3.0) or 3.0)
    raster_ops = [
        f"逐页打开 pdfplumber，共 {len(pages)} 页；页图 resolution = 72 × zoom({zoom}) = {int(72 * zoom)} dpi",
        "抽字符：dedupe_chars() 去重 → _has_color() 剔除白字/浅色水印",
        "词间空格：关闭（insert_word_spaces=False），英文数字不额外插空格",
        f"OCR 补字：pdfplumber 抽不到字时补；本次命中 {ocr_pages} 页",
        "页图只供版面模型用，算完即释放，仅留宽高元数据",
    ]
    layout_ops = [
        "逐页选后端（auto 按字符密度）：< 20 字 → yolox；≥ 120 字 → layout；其余 → hybrid",
        "layout：DeepDoc LayoutRecognizer 在页图上出区域（title/text/table/figure）",
        "yolox：unstructured YOLOX 只看页图出框，再换算回 PDF 坐标",
        "hybrid：DeepDoc 文字区 + YOLOX 图区合并",
        "score 阈值 0.25；区域按 (top, x0) 排序",
        "文本行与区域重叠 ≥ 15% 才挂 layout_type（table/figure 除外）",
        "上下行拼接：关闭（with_text_concat=False）",
    ]
    phases = [
        {
            "id": "pages",
            "title": "① 按页转图 + 识别",
            "aim": "把每页变成「页图 + 字符坐标」两路输入，给后面的版面模型吃",
            "branches": [
                {
                    "cond": "字多（电子版）",
                    "handler": "pdfplumber 抽字符",
                    "note": "抽字符坐标，只留有颜色的字，按字号/行距聚成行",
                },
                {
                    "cond": "字太少（扫描页）",
                    "handler": "OCR 补字",
                    "note": "先 DeepDoc OCR，没结果再上本机 Tesseract",
                },
                {
                    "cond": "两路都有结果",
                    "handler": "合并去重",
                    "note": "重叠（IoU>0.5）时只保留电子字，避免双份",
                },
            ],
            "facts": [
                ["页数", f"{len(pages)} 页"],
                ["页图", f"{int(72 * zoom)} dpi（72 × zoom {zoom}）"],
                ["电子字", str(native_n)],
                ["OCR 补字", f"{ocr_n} 字 · 命中 {ocr_pages} 页"],
                ["词间空格", "不插（insert_word_spaces=False）"],
            ],
        },
        {
            "id": "layout",
            "title": "② 版面识别（判行 + 判区）",
            "aim": "先用 pdfplumber 把字符聚成行，再给每行/每区贴版面类型标签",
            "branches": [
                {
                    "cond": "字符 ≥ 120（电子版）",
                    "handler": "DeepDoc layout.onnx",
                    "note": "行文字 + 坐标一起丢模型，逐行返回类型；认得标题/正文/表/图",
                },
                {
                    "cond": "字符 20–119（图文混排）",
                    "handler": "hybrid：DeepDoc 管文字和表 + YOLOX 只补图",
                    "note": "DeepDoc 出文字区，YOLOX 在页图上补图区，再合并",
                },
                {
                    "cond": "字符 < 20（扫描件）",
                    "handler": "YOLOX 只看页图",
                    "note": "只认图不认字，框出的区域整体当一个标签",
                },
            ],
            "facts": [
                ["本文件走的后端", backend_txt],
                ["识别到的区域", label_txt],
                ["文本行", str(line_n)],
                ["能认的类别", "title / text / reference / figure / figure caption / table / table caption / equation"],
                ["行贴标签门槛", "与区域重叠 ≥ 15%（table / figure 除外）"],
                ["上下行拼接", "关闭"],
            ],
        },
    ]
    stages = [
        {
            "id": "raster",
            "title": "转图 + 抽字符",
            "ok": bool(pages),
            "detail": f"{len(pages)} 页 · 字符 {char_n} · OCR 补字页 {ocr_pages} · 不插词间空格",
            "ops": raster_ops,
        },
        {
            "id": "layout",
            "title": "版面识别",
            "ok": bool(layouts),
            "detail": f"backend={backend_txt} · 区域 {label_txt} · 文本行 {line_n} · 上下行拼接关闭",
            "ops": layout_ops,
        },
        {
            "id": "filter",
            "title": "过滤页眉页脚/目录脏行",
            "ok": True,
            "detail": "content 管线：header/footer/页码/来源行；TSR 表格=" + ("开" if with_tsr else "关"),
            "ops": [
                "去 header/footer/reference 类区域内的行",
                "去页边纯页码行、图表来源行（数据来源：…）",
                "目录点线脏页与过窄旁注碎片在后一步剔除",
                "TSR（表结构识别）：本次=" + ("开，MinerU 出 HTML / pdfplumber 抽线框" if with_tsr else "关，表格按普通块处理"),
            ],
        },
        {
            "id": "blocks",
            "title": "阅读序 content_blocks",
            "ok": bool(blocks),
            "detail": f"{len(blocks)} 块 · {kind_txt}（title/text/table/figure）",
            "ops": [
                "按页面阅读顺序拼正文/标题/表格/图片",
                "表格转文本、图片留 caption，空块跳过",
            ],
        },
        {
            "id": "split_rule",
            "title": "切块规则（尚未切）",
            "ok": True,
            "detail": "下一刀：遇到 title（或目录行）flush，标题带着后续正文/表/图直到下一个标题",
            "ops": [
                "content_blocks_to_chunk_records：遇 title / 目录行就 flush 上一块",
                "每块保留 page 与 bbox，写入前查 clean_extracted_text 清洗",
            ],
        },
    ]
    return {
        "stages": stages,
        "phases": phases,
        "pages": len(pages),
        "chars": char_n,
        "native_chars": native_n,
        "ocr_chars": ocr_n,
        "ocr_pages": ocr_pages,
        "region_labels": labels,
        "line_boxes": line_n,
        "block_kinds": kinds,
        "samples": samples,
        "page_rows": page_rows,
        "tsr": with_tsr,
        "text_concat": False,
    }


def ingest_local_pdf(
    path: str | Path,
    *,
    replace_same_name: bool = True,
    with_tsr: bool = False,
    engine: str = "deepdoc",
) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    eng = (engine or "deepdoc").strip().lower()
    if eng == "mineru":
        ensure_uvgraph_root()
        from algorithm.chunk_config import get_chunk_config
        from src.configs.config_loader import get_yaml_config
        from src.knowledge.pdf_extract import mineru_markdown, mineru_title_chunks

        cfg = (get_yaml_config() or {}).get("mineru") or {}
        ingest_cfg = (get_yaml_config() or {}).get("doc_ingest") or {}
        layout = str(cfg.get("layout") or "v2")
        layout_filter = bool(ingest_cfg.get("mineru_layout_filter", True))
        chunk_by_title = bool(ingest_cfg.get("mineru_chunk_by_title", True))
        max_chars = int(get_chunk_config().get("local_max_chars") or 1400) if chunk_by_title else _chunk_chars()
        overlap = _chunk_overlap()
        chunk_rows: list[dict[str, Any]] = []
        if layout_filter and chunk_by_title:
            chunk_rows = mineru_title_chunks(
                file_path,
                layout=layout,
                max_chars=max_chars,
                overlap=overlap,
            )
        pieces = [str(row.get("content") or "").strip() for row in chunk_rows if str(row.get("content") or "").strip()]
        if not pieces:
            md = mineru_markdown(file_path, layout=layout, layout_filter=layout_filter)
            text = (md or "").strip()
            if not text:
                raise RuntimeError(f"MinerU 未产出 Markdown（检查 config.mineru 权重与 device）: {file_path}")
            pieces = _heading_chunks(text)
        else:
            text = "\n\n".join(pieces)
        if not pieces:
            raise RuntimeError(f"MinerU 文本无法切块: {file_path}")
        pages = _pdf_page_count(file_path) or max(1, len(text) // 2000)
        preprocess = {
            "pages": pages,
            "native_chars": len(text),
            "ocr_chars": 0,
            "ocr_pages": 0,
            "stages": [
                {
                    "id": "mineru",
                    "title": f"MinerU pipeline（layout={layout}）",
                    "ok": True,
                    "detail": f"共 {len(pieces)} 段 · 字符 {len(text)}",
                    "ops": [
                        "进程内 do_parse",
                        "仅保留版面：标题/正文/表格/图片（页眉页脚页码等丢弃）",
                        "按版面标题（含段落标题）+ 其后正文/表/图合并为块，过长再按字数切",
                    ],
                }
            ],
            "block_kinds": {"text": len(pieces)},
            "samples": [{"type": "text", "page": 1, "text": pieces[0][:120]}] if pieces else [],
            "page_rows": [
                {
                    "page": 1,
                    "chars": len(text),
                    "backend": "mineru",
                    "regions": 0,
                    "lines": 0,
                    "native": len(text),
                    "ocr": 0,
                }
            ],
        }
        if chunk_rows:
            records = [
                {
                    "content": str(row.get("content") or ""),
                    "page": int(row.get("page") or 1),
                    "bbox": [0, idx * 20, 100, idx * 20 + 10],
                }
                for idx, row in enumerate(chunk_rows)
            ]
        else:
            records = [
                {"content": piece, "page": 1, "bbox": [0, idx * 20, 100, idx * 20 + 10]}
                for idx, piece in enumerate(pieces)
            ]
        return _commit_records(
            file_path,
            records,
            source="mineru",
            replace_same_name=replace_same_name,
            preprocess=preprocess,
        )
    if eng == "pypdf":
        text, src = _read_pdf_text(file_path)
        pieces = _naive_chunks(text)
        if not pieces:
            raise RuntimeError(f"pypdf 未抽出可切块文本: {file_path}")
        records = [
            {"content": piece, "page": 1, "bbox": [0, idx * 20, 100, idx * 20 + 10]}
            for idx, piece in enumerate(pieces)
        ]
        preprocess = {
            "pages": max(1, len(text) // 2000),
            "native_chars": len(text),
            "ocr_chars": 0,
            "ocr_pages": 0,
            "stages": [
                {
                    "id": "pypdf",
                    "title": "pypdf 抽字 + 标题切块",
                    "ok": True,
                    "detail": f"共 {len(pieces)} 段 · 字符 {len(text)}",
                    "ops": ["pypdf 抽文本", "失败时走 report_ingest 版面", "按标题或固定长度切块"],
                }
            ],
            "block_kinds": {"text": len(pieces)},
            "samples": [{"type": "text", "page": 1, "text": pieces[0][:120]}] if pieces else [],
            "page_rows": [{"page": 1, "chars": len(text), "backend": src, "regions": 0, "lines": 0, "native": len(text), "ocr": 0}],
        }
        return _commit_records(
            file_path,
            records,
            source=src,
            replace_same_name=replace_same_name,
            preprocess=preprocess,
        )
    if eng == "unstructured":
        from algorithm.unstructured_tables import extract_tables_unstructured

        packed = extract_tables_unstructured(file_path)
        records = list(packed.get("records") or [])
        if not records:
            raise RuntimeError(packed.get("error") or f"Unstructured 没有得到文本块: {file_path}")
        return _commit_records(
            file_path,
            records,
            source="unstructured_hi_res",
            replace_same_name=replace_same_name,
            preprocess=packed.get("preprocess") if isinstance(packed.get("preprocess"), dict) else {},
        )
    ensure_uvgraph_root()
    from parser.content_pipeline import content_blocks_to_chunk_records
    from parser.pdf_parser import parse_pdf

    parsed = parse_pdf(
        file_path,
        with_layout=True,
        with_tsr=with_tsr,
        with_text_concat=False,
        with_content=True,
        retain_images=False,
        insert_word_spaces=False,
    )
    preprocess = describe_pdf_preprocess(parsed, with_tsr=with_tsr)
    records = content_blocks_to_chunk_records(parsed.content_blocks)
    if not records:
        raise RuntimeError(f"版面识别没有得到文本块: {file_path}")
    source = "deepdoc_tsr" if with_tsr else "deepdoc_layout"
    return _commit_records(
        file_path,
        records,
        source=source,
        replace_same_name=replace_same_name,
        preprocess=preprocess,
    )


def _commit_records(
    file_path: Path,
    records: list[dict[str, Any]],
    *,
    source: str,
    replace_same_name: bool,
    preprocess: dict[str, Any],
) -> dict[str, Any]:
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_local_chunks()
    name = file_path.name
    if replace_same_name:
        existing = [row for row in existing if str(row.get("doc") or "") != name]
    for idx, row in enumerate(records):
        existing.append(
            {
                "id": f"{name}:{idx}",
                "doc": name,
                "ord": idx,
                "content": row.get("content") or "",
                "source": source,
                "page": int(row.get("page") or 1),
                "bbox": row.get("bbox") or [0, idx * 20, 100, idx * 20 + 10],
            }
        )
    with _CHUNKS.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    docs = list_local_documents()
    _MANIFEST.write_text(json.dumps({"documents": docs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "name": name,
        "source": source,
        "chunk_count": len(records),
        "total_chunks": len(existing),
        "store": str(_CHUNKS),
        "preprocess": preprocess,
    }


def append_local_text_chunks(
    title: str,
    text: str,
    *,
    source: str = "knowledge_pipe",
    replace_same_name: bool = True,
) -> dict[str, Any]:
    """将纯文本切块写入本地手册库（RAGFlow 未起时的检索源）。"""
    name = re.sub(r"\s+", " ", (title or "snippet").strip()) or "snippet"
    body = (text or "").strip()
    if not body:
        raise ValueError("text 为空")
    existing = load_local_chunks()
    if replace_same_name:
        existing = [row for row in existing if str(row.get("doc") or "") != name]
    size = max(120, _chunk_chars())
    overlap = min(_chunk_overlap(), size // 4)
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        chunk = para.strip()
        if not chunk:
            continue
        start = 0
        while start < len(chunk):
            parts.append(chunk[start : start + size])
            if start + size >= len(chunk):
                break
            start += max(1, size - overlap)
    if not parts:
        parts = [body[:size]]
    base_ord = len(existing)
    for idx, content in enumerate(parts):
        existing.append(
            {
                "id": f"{name}:{base_ord + idx}",
                "doc": name,
                "ord": base_ord + idx,
                "content": content,
                "source": source,
                "page": 1,
                "bbox": [0, idx * 20, 100, idx * 20 + 10],
            }
        )
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    with _CHUNKS.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    docs = list_local_documents()
    _MANIFEST.write_text(json.dumps({"documents": docs}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "doc": name,
        "source": source,
        "chunk_count": len(parts),
        "total_chunks": len(existing),
        "store": str(_CHUNKS),
    }


def load_local_chunks() -> list[dict[str, Any]]:
    if not _CHUNKS.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in _CHUNKS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def list_local_documents() -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in load_local_chunks():
        name = str(row.get("doc") or "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return [{"name": name, "chunk_count": count, "run": "DONE"} for name, count in counts.items()]


def retrieve_local(question: str, *, top_k: int = 6) -> list[str]:
    return [str(row.get("text") or "") for row in retrieve_local_hits(question, top_k=top_k)]


def retrieve_local_hits(question: str, *, top_k: int = 6) -> list[dict[str, Any]]:
    query = (question or "").strip()
    if not query:
        return []
    tokens = _query_terms(query)
    scored: list[tuple[int, str, str]] = []
    for row in load_local_chunks():
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        score = sum(content.count(token) for token in tokens)
        if score <= 0:
            continue
        doc = str(row.get("doc") or "")
        text = f"{doc}：{content}" if doc else content
        scored.append((score, text[:400], doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    hits: list[dict[str, Any]] = []
    for idx, (score, text, doc) in enumerate(scored[:top_k], start=1):
        hits.append(
            {
                "text": text,
                "doc": doc,
                "score": float(score),
                "recall_rank": idx,
                "channel": "local",
            }
        )
    return hits


def _query_terms(query: str) -> list[str]:
    compact = re.sub(r"\s+", "", query)
    terms: list[str] = []
    if compact:
        terms.append(compact)
        for size in (4, 3, 2):
            if len(compact) < size:
                continue
            terms.extend(compact[i : i + size] for i in range(0, len(compact) - size + 1))
    return list(dict.fromkeys(item for item in terms if len(item) >= 2))[:40]


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:
        return 0


def mineru_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mineru.cli.common") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _read_pdf_text(path: Path) -> tuple[str, str]:
    pages: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page in reader.pages:
            pages.append((page.extract_text() or "").strip())
        text = "\n".join(item for item in pages if item)
        if len(text) >= 80:
            return text, "pypdf"
    except Exception:
        logger.warning("pypdf extract failed", exc_info=True)
    try:
        from algorithm.report_ingest import _extract_markdown

        parsed = _extract_markdown(path)
        markdown = str(getattr(parsed, "markdown", "") or "")
        if markdown.strip():
            return markdown, str(getattr(parsed, "source", "") or "pdf_extract")
    except Exception:
        logger.warning("layout extract failed", exc_info=True)
    if pages:
        return "\n".join(pages), "pypdf"
    raise RuntimeError(f"无法抽出文字: {path}")


def _heading_chunks(text: str) -> list[str]:
    try:
        from parser.text_clean import split_by_headings

        pieces = split_by_headings(text)
        return pieces or _naive_chunks(text)
    except (ImportError, ModuleNotFoundError):
        return _naive_chunks(text)


def _naive_chunks(text: str) -> list[str]:
    cleaned = re.sub(r"\r\n", "\n", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    paras = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks: list[str] = []
    buf = ""
    limit = _chunk_chars()
    overlap = _chunk_overlap()
    for para in paras:
        if len(para) > limit * 2:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            step = max(1, limit - overlap)
            for start in range(0, len(para), step):
                piece = para[start : start + limit].strip()
                if piece:
                    chunks.append(piece)
            continue
        if buf and len(buf) + 1 + len(para) > limit:
            chunks.append(buf.strip())
            tail = buf[-overlap:] if len(buf) > overlap else buf
            buf = (tail + "\n" + para).strip()
        else:
            buf = f"{buf}\n{para}".strip() if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or ([cleaned[:limit]] if cleaned else [])
