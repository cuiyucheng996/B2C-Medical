"""公共知识评测台：耗时展示统一为秒（内部仍用毫秒计时）。"""

from __future__ import annotations

from typing import Any

__all__ = ["ms_to_sec_text", "numbered_lines", "format_query_plan"]


def ms_to_sec_text(ms: Any) -> str:
    if ms is None or ms == "" or ms == "-":
        return "-"
    try:
        val = float(ms)
    except (TypeError, ValueError):
        return str(ms)
    sec = val / 1000.0
    if sec >= 10:
        return f"{sec:.2f} s"
    if sec >= 1:
        return f"{sec:.2f} s"
    return f"{sec:.3f} s"


def numbered_lines(
    items: list[Any],
    *,
    max_items: int = 12,
    max_chars: int = 280,
    empty: str = "（无）",
) -> str:
    rows: list[str] = []
    for idx, raw in enumerate(items[:max_items], start=1):
        text = str(raw or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        rows.append(f"{idx}. {text}")
    return "\n".join(rows) if rows else empty


def format_query_plan(
    *,
    seeds: list[str],
    query_text: str,
    main_query: str = "",
    expand_queries: list[str] | None = None,
    multi_query: bool | None = None,
    max_queries: int | None = None,
) -> str:
    """评测台：seeds / 主查询条目 / 拼合串 / 扩展检索句，均带序号。"""
    q = (query_text or "").strip()
    seed_items = [str(s).strip() for s in seeds if str(s).strip()]
    needles = list(dict.fromkeys([*seed_items, q])) if q or seed_items else seed_items
    parts = [
        "【seeds】",
        numbered_lines(seed_items),
        "",
        "【主查询条目（seeds + 问句，去重后按序拼入）】",
        numbered_lines(needles),
    ]
    merged = (main_query or "").strip() or " ".join(item for item in needles if item)
    if merged:
        show = merged if len(merged) <= 240 else merged[:239] + "…"
        parts.extend(["", "【拼合后检索串】", show])
    expanded = [str(item).strip() for item in (expand_queries or []) if str(item).strip()]
    if expanded:
        head = "【待检索句（expand_queries）】"
        if multi_query is not None or max_queries is not None:
            head += f"（multi_query={multi_query} max={max_queries}，共 {len(expanded)} 条）"
        parts.extend(["", head, numbered_lines(expanded)])
    return "\n".join(parts)
