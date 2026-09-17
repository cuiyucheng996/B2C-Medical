"""C 端文档知识库：调用 RAGFlow HTTP API。分块展示走 RAGFlow Web，不另做前端。"""

from __future__ import annotations

import json
import logging
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from configs.paths_b2c import ensure_uvgraph_root
from algorithm.pdf_kb import ingest_local_pdf, list_local_documents, retrieve_local, retrieve_local_hits

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_dataset",
    "kb_status",
    "list_documents",
    "replace_pdf",
    "retrieve_chunks",
    "retrieve_hits",
    "ragflow_configured",
]


def ragflow_configured() -> bool:
    cfg = _cfg()
    return bool(cfg.get("base_url") and cfg.get("api_key"))


def kb_status() -> dict[str, Any]:
    cfg = _cfg()
    ping = ping_ragflow()
    dataset = None
    if ping.get("ok"):
        try:
            dataset = ensure_dataset()
        except Exception as exc:
            ping = {"ok": False, "error": str(exc)}
    local_docs = list_local_documents()
    return {
        "configured": ragflow_configured(),
        "reachable": bool(ping.get("ok")),
        "base_url": cfg.get("base_url") or "",
        "web_url": cfg.get("web_url") or cfg.get("base_url") or "",
        "dataset_name": cfg.get("dataset_name") or "b2c_exam_handbook",
        "dataset": dataset,
        "local_documents": local_docs,
        "error": ping.get("error") or "",
        "hint": "RAGFlow 未启动时使用本地分块；启动后可在官方 Web 看块。",
    }


def ping_ragflow() -> dict[str, Any]:
    if not ragflow_configured():
        return {"ok": False, "error": "未配置 RAGFLOW_API_KEY 或 ragflow.base_url"}
    try:
        payload = _request("GET", "/api/v1/datasets", query={"page": 1, "page_size": 1}, timeout=8)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    if int(payload.get("code") or 0) != 0:
        return {"ok": False, "error": str(payload.get("message") or payload)}
    return {"ok": True}


def ensure_dataset() -> dict[str, Any]:
    cfg = _cfg()
    dataset_id = str(cfg.get("dataset_id") or "").strip()
    name = str(cfg.get("dataset_name") or "b2c_exam_handbook").strip()
    if dataset_id:
        found = _request("GET", "/api/v1/datasets", query={"id": dataset_id, "page": 1, "page_size": 1})
        rows = _rows(found)
        if rows:
            return rows[0]
    listed = _request("GET", "/api/v1/datasets", query={"name": name, "page": 1, "page_size": 10})
    for row in _rows(listed):
        if str(row.get("name") or "") == name or str(row.get("id") or "") == dataset_id:
            return row
    from algorithm.chunk_config import get_chunk_config

    chunk = get_chunk_config()
    created = _request(
        "POST",
        "/api/v1/datasets",
        json_body={
            "name": name,
            "description": "B2C 公共知识：检查手册等 PDF，可替换更新",
            "chunk_method": str(chunk.get("chunk_method") or cfg.get("chunk_method") or "naive"),
            "parser_config": {
                "chunk_token_num": int(chunk.get("chunk_token_num") or 256),
                "delimiter": str(chunk.get("delimiter") or "\n"),
                "layout_recognize": str(chunk.get("layout_recognize") or "Plain Text"),
            },
        },
    )
    data = created.get("data") if isinstance(created.get("data"), dict) else {}
    if int(created.get("code") or 0) != 0 or not data:
        raise RuntimeError(created.get("message") or "创建 RAGFlow 知识库失败")
    return data


def list_documents() -> list[dict[str, Any]]:
    if ping_ragflow().get("ok"):
        dataset = ensure_dataset()
        dataset_id = str(dataset.get("id") or "")
        payload = _request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents",
            query={"page": 1, "page_size": 50},
        )
        return _rows(payload)
    return list_local_documents()


def retrieve_chunks(question: str, *, top_k: int = 6) -> list[str]:
    return [str(row.get("text") or "") for row in retrieve_hits(question, top_k=top_k) if row.get("text")]


def retrieve_hits(question: str, *, top_k: int = 6) -> list[dict[str, Any]]:
    """召回带分：优先 RAGFlow 返回序，失败走本地词频。"""
    text = (question or "").strip()
    if not text:
        return []
    if ping_ragflow().get("ok"):
        remote = _retrieve_remote_hits(text, top_k=top_k)
        if remote:
            return remote
    return retrieve_local_hits(text, top_k=top_k)


def _retrieve_remote(text: str, *, top_k: int) -> list[str]:
    return [str(row.get("text") or "") for row in _retrieve_remote_hits(text, top_k=top_k)]


def _retrieve_remote_hits(text: str, *, top_k: int) -> list[dict[str, Any]]:
    try:
        dataset = ensure_dataset()
        dataset_id = str(dataset.get("id") or "")
        payload = _request(
            "POST",
            "/api/v1/retrieval",
            json_body={
                "question": text,
                "dataset_ids": [dataset_id],
                "page": 1,
                "page_size": max(1, top_k),
                "similarity_threshold": 0.2,
                "vector_similarity_weight": 0.3,
                "keyword": True,
            },
            timeout=20,
        )
    except Exception:
        logger.warning("ragflow retrieve failed", exc_info=True)
        return []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    raw_chunks = data.get("chunks") if isinstance(data, dict) else []
    if not isinstance(raw_chunks, list):
        raw_chunks = payload.get("data") if isinstance(payload.get("data"), list) else []
    hits: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_chunks or [], start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("content_with_weight") or "").strip()
        doc = str(item.get("document_keyword") or item.get("docnm_kwd") or "").strip()
        if not content:
            continue
        prefix = f"{doc}：" if doc else ""
        score = item.get("similarity")
        if score is None:
            score = item.get("vector_similarity")
        if score is None:
            score = item.get("term_similarity")
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        hits.append(
            {
                "text": (prefix + content)[:400],
                "doc": doc,
                "score": round(score_f, 4),
                "recall_rank": idx,
                "channel": "ragflow",
            }
        )
        if len(hits) >= top_k:
            break
    return hits


def replace_pdf(
    path: str | Path,
    *,
    replace_same_name: bool = True,
    chunk_source: str = "local",
    with_tsr: bool = False,
) -> dict[str, Any]:
    """先写入本地分块；RAGFlow 可用时再同步过去。同名默认覆盖。

    chunk_source=local：DeepDoc 版面模型认 title/text，再按 title 分块上传（不开表格 TSR）。--tsr 才加表结构。
    chunk_source=ragflow：上传后让 RAGFlow 自己解析（旧行为）。
    """
    local = ingest_local_pdf(path, replace_same_name=replace_same_name, with_tsr=with_tsr)
    result: dict[str, Any] = {"local": local, "ragflow": None}
    if not ping_ragflow().get("ok"):
        return result
    file_path = Path(path)
    dataset = ensure_dataset()
    dataset_id = str(dataset.get("id") or "")
    filename = file_path.name
    if replace_same_name:
        stale_ids = [
            str(row.get("id") or "")
            for row in _rows(
                _request(
                    "GET",
                    f"/api/v1/datasets/{dataset_id}/documents",
                    query={"page": 1, "page_size": 50},
                )
            )
            if str(row.get("name") or "") == filename and row.get("id")
        ]
        if stale_ids:
            _request(
                "DELETE",
                f"/api/v1/datasets/{dataset_id}/documents",
                json_body={"ids": stale_ids},
            )
    uploaded = _upload_file(dataset_id, file_path)
    docs = uploaded.get("data") if isinstance(uploaded.get("data"), list) else []
    if int(uploaded.get("code") or 0) != 0 or not docs:
        raise RuntimeError(uploaded.get("message") or "上传 RAGFlow 失败")
    doc_id = str(docs[0].get("id") or "")
    source = (chunk_source or "local").strip().lower()
    if source == "ragflow":
        parsed = _request(
            "POST",
            f"/api/v1/datasets/{dataset_id}/chunks",
            json_body={"document_ids": [doc_id]},
        )
        if int(parsed.get("code") or 0) != 0:
            raise RuntimeError(parsed.get("message") or "触发解析失败")
        status = _wait_parse(dataset_id, doc_id)
        result["ragflow"] = {
            "dataset_id": dataset_id,
            "document_id": doc_id,
            "name": filename,
            "chunk_source": "ragflow",
            "parse": status,
            "web_url": _cfg().get("web_url") or "",
        }
        return result

    from algorithm.pdf_kb import load_local_chunks

    rows = [
        row
        for row in load_local_chunks()
        if str(row.get("doc") or "") == filename and str(row.get("content") or "").strip()
    ]
    rows.sort(key=lambda item: int(item.get("ord") or 0))
    if not rows:
        texts = _parse_local_chunks(file_path, with_tsr=with_tsr)
        rows = [{"content": text, "ord": i, "page": 1, "bbox": [0, i * 20, 100, i * 20 + 10]} for i, text in enumerate(texts)]
    pushed = 0
    for row in rows:
        text = str(row.get("content") or "").strip()
        added: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(3):
            try:
                added = _request(
                    "POST",
                    f"/api/v1/datasets/{dataset_id}/documents/{doc_id}/chunks",
                    json_body={
                        "content": text,
                        "important_keywords": _chunk_keywords(text, pushed),
                        "positions": [_ragflow_position(row, pushed)],
                    },
                )
            except RuntimeError as exc:
                last_error = str(exc)
                time.sleep(2 * (attempt + 1))
                continue
            break
        if added is None or int(added.get("code") or 0) != 0:
            raise RuntimeError((added or {}).get("message") or last_error or "写入 RAGFlow 分块失败")
        pushed += 1
    result["ragflow"] = {
        "dataset_id": dataset_id,
        "document_id": doc_id,
        "name": filename,
        "chunk_source": "local",
        "chunk_count": pushed,
        "web_url": _cfg().get("web_url") or "",
    }
    return result


def _ragflow_position(row: dict[str, Any], index: int) -> list[int]:
    """RAGFlow: [page, left, right, top, bottom]，按页码和纵坐标从头排到尾。"""
    page = max(1, int(row.get("page") or 1))
    bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else []
    if len(bbox) >= 4:
        left, top, right, bottom = (int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3])))
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        return [page, left, right, top, bottom]
    top = index * 20
    return [page, 0, 100, top, top + 10]


def _chunk_keywords(text: str, index: int) -> list[str]:
    title = (text or "").split("\n", 1)[0].strip()[:40]
    keys = [f"{index + 1:04d}"]
    if title:
        keys.append(title)
    return keys


def _parse_local_chunks(file_path: Path, *, with_tsr: bool) -> list[str]:
    ensure_uvgraph_root()
    from parser.content_pipeline import content_blocks_to_chunk_texts
    from parser.pdf_parser import parse_pdf

    parsed = parse_pdf(
        file_path,
        with_tsr=with_tsr,
        with_text_concat=False,
        with_content=True,
        retain_images=False,
        insert_word_spaces=False,
    )
    chunks = content_blocks_to_chunk_texts(parsed.content_blocks)
    if not chunks:
        raise RuntimeError("本地解析没有得到可写入的文本块")
    return chunks


def _wait_parse(dataset_id: str, document_id: str, *, attempts: int = 40) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for _ in range(attempts):
        payload = _request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents",
            query={"id": document_id, "page": 1, "page_size": 1},
        )
        rows = _rows(payload)
        last = rows[0] if rows else {}
        run = str(last.get("run") or last.get("status") or "").upper()
        if run in {"DONE", "SUCCESS", "2", "COMPLETE", "COMPLETED"}:
            return last
        if run in {"FAIL", "FAILED", "CANCEL", "CANCELLED"}:
            raise RuntimeError(f"RAGFlow 解析失败: {last}")
        time.sleep(3)
    return last


def _cfg() -> dict[str, Any]:
    ensure_uvgraph_root()
    from src.configs import RAGFLOW_CONFIG, get_app_settings

    settings = get_app_settings()
    return {
        "base_url": str(RAGFLOW_CONFIG.get("base_url") or "http://127.0.0.1:9380").rstrip("/"),
        "web_url": str(RAGFLOW_CONFIG.get("web_url") or "http://127.0.0.1:8088").rstrip("/"),
        "dataset_name": RAGFLOW_CONFIG.get("dataset_name") or "b2c_exam_handbook",
        "dataset_id": RAGFLOW_CONFIG.get("dataset_id") or "",
        "chunk_method": RAGFLOW_CONFIG.get("chunk_method") or "naive",
        "timeout": float(RAGFLOW_CONFIG.get("timeout_seconds") or 180),
        "api_key": (settings.ragflow_api_key or "").strip(),
    }


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        docs = data.get("docs") or data.get("datasets") or data.get("kbs") or []
        if isinstance(docs, list):
            return [item for item in docs if isinstance(item, dict)]
    return []


def _request(
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    cfg = _cfg()
    if not cfg["api_key"]:
        raise RuntimeError("未配置 RAGFLOW_API_KEY")
    url = cfg["base_url"] + path
    if query:
        url += "?" + urlencode({k: v for k, v in query.items() if v is not None and v != ""})
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Accept": "application/json",
    }
    data = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method.upper())
    return _read(req, timeout=timeout if timeout is not None else cfg["timeout"])


def _upload_file(dataset_id: str, file_path: Path) -> dict[str, Any]:
    cfg = _cfg()
    boundary = uuid.uuid4().hex
    filename = file_path.name
    mime = mimetypes.guess_type(filename)[0] or "application/pdf"
    file_bytes = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    url = f"{cfg['base_url']}/api/v1/datasets/{dataset_id}/documents"
    req = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    return _read(req, timeout=max(cfg["timeout"], 300))


def _read(req: Request, *, timeout: float) -> dict[str, Any]:
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"message": raw or str(exc)}
        raise RuntimeError(payload.get("message") or str(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"连不上 RAGFlow: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(raw[:300]) from exc
    if isinstance(payload, dict):
        return payload
    return {"data": payload}
