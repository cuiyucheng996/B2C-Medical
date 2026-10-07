"""SSE 流式输出工具（医疗 API / 对话共用）。"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "chunk_text",
    "sse_payload",
]


def sse_payload(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def chunk_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)
