"""评测 / 运维：知识片段写入本地库（可选 RAGFlow 仅作检索通道，文本块以 local 为主）。"""

from __future__ import annotations

import logging
import re
from typing import Any

from algorithm.pdf_kb import append_local_text_chunks
from algorithm.ragflow_kb import ping_ragflow, ragflow_configured

logger = logging.getLogger(__name__)

__all__ = ["ingest", "pipe_status"]


def pipe_status() -> dict[str, Any]:
    sinks = ["local"]
    ragflow_ready = False
    if ragflow_configured():
        sinks.append("ragflow")
        ragflow_ready = bool(ping_ragflow().get("ok"))
    return {
        "sinks": sinks,
        "ragflow_ready": ragflow_ready,
    }


def _fetch_url_text(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("url 为空")
    from urllib.request import Request, urlopen

    req = Request(raw, headers={"User-Agent": "B2C-KnowledgePipe/1.0"})
    with urlopen(req, timeout=20) as resp:
        data = resp.read()
    try:
        return data.decode("utf-8", errors="ignore").strip()
    except Exception:
        return data.decode("latin-1", errors="ignore").strip()


def ingest(
    *,
    text: str = "",
    title: str = "",
    query: str = "",
    url: str = "",
) -> dict[str, Any]:
    """写入知识片段：优先 text；否则 query 联网摘要；否则 url 抓正文。"""
    name = (title or "").strip() or "knowledge-snippet"
    body = (text or "").strip()
    if not body and (query or "").strip():
        try:
            from algorithm.dashscope_web_search import dashscope_web_search

            hit = dashscope_web_search((query or "").strip())
            body = str(hit.get("answer") or hit.get("content") or "").strip()
            if not name or name == "knowledge-snippet":
                name = re.sub(r"\s+", " ", (query or "").strip())[:80] or name
        except Exception as exc:
            raise RuntimeError(f"联网搜索失败: {exc}") from exc
    if not body and (url or "").strip():
        try:
            body = _fetch_url_text(url)
            if not name or name == "knowledge-snippet":
                name = re.sub(r"[^\w.\u4e00-\u9fff-]+", "_", (url or "").split("/")[-1] or "web-page")[:80]
        except Exception as exc:
            raise RuntimeError(f"抓取 url 失败: {exc}") from exc
    if not body:
        raise ValueError("需要 text，或可用的 query / url")
    local = append_local_text_chunks(name, body, source="knowledge_pipe", replace_same_name=True)
    out: dict[str, Any] = {"ok": True, "local": local, "title": name}
    try:
        if ragflow_configured():
            rf_ok = bool(ping_ragflow().get("ok"))
            out["ragflow"] = {
                "ok": rf_ok,
                "note": "文本块已写入 local；ragflow 通道检索已有 RAGFlow 数据集",
            }
            out["ragflow_ready"] = rf_ok
    except Exception as exc:
        logger.debug("ragflow status skipped: %s", exc)
    return out
