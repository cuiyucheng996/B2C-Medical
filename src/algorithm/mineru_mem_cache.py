"""进程内 MinerU 拆表结果缓存（按上传文件字节 SHA256）。explain 写入，再次上传同内容命中。"""

from __future__ import annotations

import hashlib
from typing import Any

_MAX_ENTRIES = 48
_STORE: dict[str, dict[str, Any]] = {}


def file_content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_mineru_mem(content_sha256: str) -> dict[str, Any] | None:
    key = (content_sha256 or "").strip()
    if not key:
        return None
    row = _STORE.get(key)
    return dict(row) if isinstance(row, dict) else None


def put_mineru_mem(
    content_sha256: str,
    *,
    structured_tables: list[dict],
    indicators: list[dict[str, str]],
    parse_source: str,
) -> None:
    key = (content_sha256 or "").strip()
    if not key:
        return
    if len(_STORE) >= _MAX_ENTRIES and key not in _STORE:
        drop = next(iter(_STORE))
        _STORE.pop(drop, None)
    _STORE[key] = {
        "structured_tables": [dict(item) for item in structured_tables],
        "indicators": [dict(item) for item in indicators],
        "parse_source": str(parse_source or ""),
    }


def mineru_payload_from_mem(content_sha256: str) -> tuple[list[dict], list[dict[str, str]], str] | None:
    row = get_mineru_mem(content_sha256)
    if not row:
        return None
    tables = list(row.get("structured_tables") or [])
    indicators = [dict(item) for item in (row.get("indicators") or []) if isinstance(item, dict)]
    if not tables and not indicators:
        return None
    source = str(row.get("parse_source") or ("mineru_table" if tables else ""))
    return tables, indicators, source
