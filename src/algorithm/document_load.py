"""C 端文档加载（预处理 ingestion 第一步）。

职责：把用户文件/网页变成统一的 LangChain Document 列表，并抽出
body / titles / tables，供 file_gate.parse_user_file 后续判定角色、抽指标。

入口只此一份。图片/扫描 PDF 仍走 report_ingest（YOLOX），不走这里。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

__all__ = [
    "LoadedSource",
    "chunks_from_documents",
    "load_user_source",
    "load_web",
    "stamp_chunks",
]

os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (uvgraph-b2c document-load)")

_TEXT_EXT = {".txt"}
_MD_EXT = {".md", ".markdown"}
_DOCX_EXT = {".docx"}
_HTML_EXT = {".html", ".htm"}
_PDF_EXT = {".pdf"}


@dataclass
class LoadedSource:
    loader: str
    documents: list[Document] = field(default_factory=list)
    body: str = ""
    titles: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    source: str = ""

    def to_parse_ns(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            source=self.loader,
            body=self.body,
            titles=self.titles,
            tables=self.tables,
            image_count=0,
            markdown="",
            documents=self.documents,
        )


def load_user_source(
    path: str | Path,
    *,
    user_id: str = "",
    url: str = "",
) -> LoadedSource:
    """按后缀分发给对应 Loader；url 非空时走网页加载。"""
    if (url or "").strip():
        return load_web(url.strip(), user_id=user_id)
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"文件不存在: {source}")
    suffix = source.suffix.lower()
    if suffix in _TEXT_EXT:
        bundle = _load_txt(source)
    elif suffix in _MD_EXT:
        bundle = _load_markdown(source)
    elif suffix in _DOCX_EXT:
        bundle = _load_docx(source)
    elif suffix in _HTML_EXT:
        bundle = _load_html(source)
    elif suffix in _PDF_EXT:
        bundle = _load_pdf_fast(source)
    else:
        raise ValueError(f"文档加载不支持该格式: {suffix or source.name}")
    _stamp_user(bundle.documents, user_id=user_id, saved_path=str(source))
    return bundle


def load_web(url: str, *, user_id: str = "", css_id: str | None = None) -> LoadedSource:
    """WebBaseLoader：抓网页。css_id 有值时只取该容器，避免导航噪声。"""
    import bs4
    from langchain_community.document_loaders import WebBaseLoader

    kwargs: dict[str, Any] = {}
    if css_id:
        kwargs["bs_kwargs"] = {"parse_only": bs4.SoupStrainer(id=css_id)}
    docs = WebBaseLoader(web_paths=(url,), **kwargs).load()
    _stamp_user(docs, user_id=user_id, saved_path=url)
    return _from_docs(docs, loader="web", source=url)


def _load_txt(path: Path) -> LoadedSource:
    from langchain_community.document_loaders import TextLoader

    docs = TextLoader(str(path), encoding="utf-8").load()
    return _from_docs(docs, loader="txt", source=str(path))


def _load_markdown(path: Path) -> LoadedSource:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader

    docs = UnstructuredMarkdownLoader(
        str(path),
        mode="elements",
        strategy="fast",
    ).load()
    return _from_docs(docs, loader="markdown", source=str(path))


def _load_docx(path: Path) -> LoadedSource:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader

    docs = UnstructuredWordDocumentLoader(
        str(path),
        mode="elements",
        strategy="fast",
    ).load()
    return _from_docs(docs, loader="docx", source=str(path))


def _load_html(path: Path) -> LoadedSource:
    from langchain_community.document_loaders import BSHTMLLoader

    docs = BSHTMLLoader(str(path), open_encoding="utf-8").load()
    return _from_docs(docs, loader="html", source=str(path))


def _load_pdf_fast(path: Path) -> LoadedSource:
    """文字型 PDF 的快路径。扫描件/化验单仍应由 report_ingest 走 YOLOX。"""
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(str(path), extraction_mode="layout").load()
    return _from_docs(docs, loader="pdf", source=str(path))


def _from_docs(docs: list[Document], *, loader: str, source: str) -> LoadedSource:
    titles: list[str] = []
    tables: list[str] = []
    bodies: list[str] = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        meta = dict(doc.metadata or {})
        category = str(meta.get("category") or "")
        html = str(meta.get("text_as_html") or "").strip()
        if html and ("<table" in html.lower() or category == "Table"):
            tables.append(html)
        if category in {"Title", "Header"} and text:
            titles.append(text)
        if text:
            bodies.append(text)
    if not titles:
        titles = [item[:40] for item in bodies[:8] if item]
    return LoadedSource(
        loader=loader,
        documents=docs,
        body="\n".join(bodies),
        titles=titles[:12],
        tables=tables[:8],
        source=source,
    )


def _stamp_user(docs: list[Document], *, user_id: str, saved_path: str) -> None:
    """加载期写入隔离字段，后面向量化/引用都靠它，事后补不回来。"""
    uid = (user_id or "").strip()
    for doc in docs:
        meta = dict(doc.metadata or {})
        if uid:
            meta["user_id"] = uid
        meta["saved_path"] = saved_path
        doc.metadata = json.loads(json.dumps(meta, ensure_ascii=False, default=str))


def stamp_chunks(
    items: list[dict[str, Any]] | None,
    *,
    user_id: str,
    saved_path: str,
) -> list[dict[str, Any]]:
    """图/PDF 版面元素、Loader 块共用：补 user_id / 路径 / 页码 / 元素类型。"""
    uid = (user_id or "").strip()
    chunks: list[dict[str, Any]] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        page = raw.get("page") if raw.get("page") is not None else raw.get("page_number")
        try:
            page_no = int(page or 1)
        except (TypeError, ValueError):
            page_no = 1
        text = str(raw.get("text") or raw.get("page_content") or "").strip()
        html = str(raw.get("html") or raw.get("text_as_html") or "").strip()
        chunks.append(
            {
                "user_id": uid,
                "saved_path": saved_path,
                "page": page_no,
                "category": str(raw.get("category") or "Unknown"),
                "text": text,
                "html": html,
            }
        )
    return chunks


def chunks_from_documents(
    docs: list[Document] | None,
    *,
    user_id: str,
    saved_path: str,
) -> list[dict[str, Any]]:
    """LangChain Document → 与版面元素同一套 chunk 字段。"""
    rows: list[dict[str, Any]] = []
    for doc in docs or []:
        meta = dict(doc.metadata or {})
        rows.append(
            {
                "category": str(meta.get("category") or "Unknown"),
                "page": meta.get("page_number") if meta.get("page_number") is not None else meta.get("page"),
                "text": (doc.page_content or "").strip(),
                "html": str(meta.get("text_as_html") or meta.get("html") or ""),
            }
        )
    return stamp_chunks(rows, user_id=user_id, saved_path=saved_path)
