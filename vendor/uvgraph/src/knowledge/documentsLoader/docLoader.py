from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src.configs.config_loader  # noqa: F401 — 从 config.yaml 加载 OCR/Poppler 路径并 apply_cache_env

from src.configs.cache import HF_HUB_CACHE_DIR

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader, WebBaseLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

_KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
_PRODUCT_DATA_DIR = Path(__file__).resolve().parents[1] / "product_data"


def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """将 metadata 转为可 JSON 序列化的 dict（处理 numpy 等类型）。"""
    return json.loads(json.dumps(metadata, ensure_ascii=False, default=str))


def documents_to_markdown(
    documents: list[Document],
    *,
    source_name: str,
) -> str:
    """将 LangChain Document 列表转为 Markdown 文本。"""
    lines = [
        f"# {source_name}",
        "",
        f"- elements: {len(documents)}",
        f"- hf_hub_cache: `{HF_HUB_CACHE_DIR}`",
        "",
    ]
    for index, doc in enumerate(documents, start=1):
        meta = _serialize_metadata(dict(doc.metadata or {}))
        category = meta.get("category", "Unknown")
        page_number = meta.get("page_number", "-")
        lines.extend(
            [
                f"## Element {index} ({category}, page {page_number})",
                "",
                doc.page_content.strip() or "_(empty)_",
                "",
                "### Metadata",
                "",
                "```json",
                json.dumps(meta, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def save_documents_to_md(
    documents: list[Document],
    *,
    source_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """将 Document 列表写入 product_data 下的 .md 文件。"""
    out_dir = output_dir or _PRODUCT_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{source_path.stem}.md"
    markdown = documents_to_markdown(documents, source_name=source_path.name)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


#文本文件加载
def text_loader():  # 返回text_documents
    text_documents = TextLoader(str(_KB_DIR / "sample.txt"), encoding="utf-8").load()
    return text_documents

#markdown文件加载
"""
UnstructuredMarkdownLoader可用于加载Markdown文件
    mode: 加载模式
        "single"    返回单个Document对象
        "elements"  按标题等元素切分文档
    strategy: 加载策略
        "fast"      快速粗粒度加载
        "hi_res"    细粒度加载，按标题层级、列表项、表格等结构细分
"""


def markdown_loader():  # 返回md_documents
    md_documents = UnstructuredMarkdownLoader(
        str(_KB_DIR / "sample.md"),
        #element根据标题等元素切分文档，single返回单个Document对象，elements返回多个Document对象
        mode="single",
        #fast快速粗粒度加载，hi_res细粒度加载，按标题层级、列表项、表格等结构细分
        strategy="fast",
    ).load()
    return md_documents


def load_pdf_file(path: str | Path, *, use_pymuPDF: bool | None = None) -> list[Document]:
    """解析任意 PDF。use_pymuPDF=True 走 PyMuPDF；False 走 Unstructured hi_res；None 先快后慢。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PDF 不存在: {source}")
    if use_pymuPDF is True:
        return PyMuPDFLoader(str(source)).load()
    if use_pymuPDF is False:
        return UnstructuredPDFLoader(
            str(source),
            mode="elements",
            # hi_res 要用到 yolox；ocr 识字；poppler 的 pdftoppm 在 .cache/external
            strategy="hi_res",
            infer_table_structure=True,
            languages=["eng", "chi_sim"],
        ).load()
    docs = PyMuPDFLoader(str(source)).load()
    text = "\n".join((doc.page_content or "") for doc in docs).strip()
    if len(text) >= 80:
        return docs
    return load_pdf_file(source, use_pymuPDF=False)


def pdf_loader(use_pymuPDF: bool = False):  # 返回pdf_documents
    source_path = _KB_DIR / "sample.pdf"
    return load_pdf_file(source_path, use_pymuPDF=use_pymuPDF)


def word_loader():  # 返回word_documents
    #它是用于加载Word文件的，它适合找title等结构
    #解析网址中的内容，比如：https://www.baidu.com/s?wd=python
    web_documents = WebBaseLoader(
        ["https://www.baidu.com/s?wd=python"],  
        bs_kwargs={"parse_only": bs4.SoupStrainer(id="article")},
    ).load()
    return web_documents

if __name__ == "__main__":
    pdf_path = _KB_DIR / "sample.pdf"
    documents = pdf_loader(use_pymuPDF=False)
    output_path = save_documents_to_md(documents, source_path=pdf_path)

