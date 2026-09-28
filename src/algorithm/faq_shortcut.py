"""完全命中的封闭话术：在进意图 / 主图之前短路。"""

from __future__ import annotations

import re

__all__ = ["match_faq_exact"]

# 只收整句相等；症状、病名不要放进来
_EXACT: dict[str, str] = {
    "你好": "greet",
    "您好": "greet",
    "在吗": "greet",
    "你好呀": "greet",
    "嗨": "greet",
    "你是谁": "who",
    "你是做什么的": "who",
    "你能做什么": "who",
    "刚才说了什么": "recall",
    "上一句说了什么": "recall",
    "我刚才说了什么": "recall",
    "我上一句说了什么": "recall",
}


def _norm(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = re.sub(r"[\s，。！？、,.!?;；:：~～]+", "", raw)
    return raw


def match_faq_exact(text: str) -> str:
    """整句命中返回话术键，否则空串。"""
    key = _norm(text)
    if not key:
        return ""
    return _EXACT.get(key, "")
