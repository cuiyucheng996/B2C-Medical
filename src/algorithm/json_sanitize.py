"""LLM 脏 JSON 容错：去围栏、截取对象、解析 dict。"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "extract_json_object_text",
    "loads_json_dict",
    "strip_markdown_json_fence",
]

_FENCE_LINE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)
_FENCE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def strip_markdown_json_fence(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    block = _FENCE_BLOCK.search(raw)
    if block:
        return (block.group(1) or "").strip()
    return _FENCE_LINE.sub("", raw).strip()


def extract_json_object_text(text: str) -> str:
    raw = strip_markdown_json_fence(text)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return ""
    return raw[start : end + 1]


def loads_json_dict(text: str) -> dict[str, Any]:
    blob = extract_json_object_text(text)
    if not blob:
        raise json.JSONDecodeError("no json object", text or "", 0)
    data = json.loads(blob)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("root not object", blob, 0)
    return data
