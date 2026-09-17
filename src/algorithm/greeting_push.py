"""进场主动推送：只根据长期记忆拼一句，不开药、不当诊断。"""

from __future__ import annotations

from typing import Any

__all__ = ["build_entry_push"]


def build_entry_push(user_id: str) -> dict[str, Any]:
    uid = (user_id or "").strip()
    if not uid:
        return {
            "text": "先填用户 id 再建会话。",
            "facts": [],
            "source": "empty_user",
        }
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from algorithm.periodic_digest import load_latest_digest, mark_digest_read

    digest = load_latest_digest(uid)
    if digest and str(digest.get("raw_text") or "").strip():
        text = str(digest.get("raw_text") or "").strip()
        mark_digest_read(str(digest.get("id") or ""))
        return {
            "text": text,
            "facts": [digest],
            "source": "daily_digest",
        }

    from src.algorithm.user_memory import load_user_memory_card

    card = load_user_memory_card(uid, limit=8)
    raw_facts = list(card.get("companion_facts") or [])
    lines: list[str] = []
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for item in raw_facts:
        text = str(item.get("fact") or item.get("name") or "").strip()
        if item.get("negated"):
            text = "已排除：" + str(item.get("name") or text)
        if not text or text.startswith("user(") or text.startswith("用户长期记忆"):
            continue
        key = text.replace(" ", "")
        if key in seen or any(key in old or old in key for old in seen):
            continue
        seen.add(key)
        lines.append(text)
        kept.append(item)
        if len(lines) >= 4:
            break
    if not lines:
        text = "还没有查到你的长期记忆。你可以直接说说近况，结束会话后我会记下来。"
        return {"text": text, "facts": kept, "source": "no_memory"}
    bullets = "\n".join(f"- {line}" for line in lines)
    text = f"进场回顾（来自你以前关场留下的记忆，不是定时报告）：\n{bullets}\n有变化直接说就行。"
    return {"text": text, "facts": kept, "source": "long_memory"}
