"""多意图子图回复合并（无二次 handoff LLM）。"""

from __future__ import annotations

from typing import Any

__all__ = ["merge_worker_replies"]


def _norm_key(text: str) -> str:
    return "".join((text or "").split())[:400]


def merge_worker_replies(rows: list[dict[str, Any]]) -> str:
    """去重后拼接各路 assistant_reply；多路时加简短意图标签便于阅读。"""
    items = [item for item in (rows or []) if isinstance(item, dict)]
    seen: set[str] = set()
    blocks: list[str] = []
    multi = len(items) > 1
    for item in items:
        body = str(item.get("reply") or "").strip()
        if not body:
            continue
        key = _norm_key(body)
        if key in seen:
            continue
        seen.add(key)
        if multi:
            label = " / ".join(
                p
                for p in (
                    str(item.get("major") or "").strip(),
                    str(item.get("intent") or "").strip(),
                )
                if p
            )
            blocks.append(f"【{label}】\n{body}" if label else body)
        else:
            blocks.append(body)
    return "\n\n".join(blocks).strip()
