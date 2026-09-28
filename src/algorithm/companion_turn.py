"""陪聊单轮进线：与 CompanionAgent 相同的 preprocess → 意图 → 抽槽对齐。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from algorithm.chat_preprocess import (
    CompanionPreprocess,
    finish_preprocess_after_intent,
    format_preprocess_for_agent,
    run_chat_preprocess,
)
from algorithm.intent_classify import (
    CompanionIntentResult,
    intent_result_to_dict,
    normalize_intent_results,
    resolve_companion_intent_items,
)
from tools.companion_tools import tools_for_intent

__all__ = [
    "CompanionTurnContext",
    "apply_file_route_to_intents",
    "attach_intent_to_preprocess",
    "build_intent_tasks",
    "resolve_companion_turn",
    "turn_result_to_payload",
]


@dataclass
class CompanionTurnContext:
    user_id: str = ""
    last_preprocess: dict[str, Any] | None = None
    last_user_text: str = ""
    slot_memory: dict[str, list[str]] | None = None
    prior_user_texts: list[str] | None = None
    recent_dialogue: str = ""
    last_file: dict[str, Any] | None = None


def apply_file_route_to_intents(
    text: str,
    items: list[CompanionIntentResult],
    *,
    last_file: dict[str, Any] | None,
) -> list[CompanionIntentResult]:
    """上传原件：打分只锁大类。化验单细意图交给 LLM；其它仍走图片/文档细类。"""
    from algorithm.file_gate import intent_for_file_role

    prefix = (text or "").lstrip()
    uploadish = prefix.startswith(("【上传检查报告】", "【上传图片】", "【上传文档】"))
    row = last_file if isinstance(last_file, dict) else {}
    if not uploadish and not row.get("saved_path") and str(row.get("role") or "") != "layout_picture":
        return items
    spec = intent_for_file_role(str(row.get("role") or ""))
    if not spec:
        return items
    if spec["handler"] == "report_upload":
        kept = [item for item in items if getattr(item, "major", "") == "报告指标解读"]
        if kept:
            return normalize_intent_results(kept)
        forced = CompanionIntentResult(
            intent="未细分",
            major="报告指标解读",
            confidence=1.0,
            source="file_gate",
            reason="；".join(list(row.get("reasons") or [])[:4]) or "文件前置分流",
            slots=(),
            handler="fallback",
        )
        return normalize_intent_results([forced])
    forced = CompanionIntentResult(
        intent=spec["intent"],
        major=spec["major"],
        confidence=1.0,
        source="file_gate",
        reason="；".join(list(row.get("reasons") or [])[:4]) or "文件前置分流",
        slots=(),
        handler=spec["handler"],
    )
    return normalize_intent_results([forced])


def build_intent_tasks(
    items: list[CompanionIntentResult],
    *,
    observe: str,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for item in items:
        tools = tools_for_intent(major=item.major, handler=item.handler)
        tasks.append(
            {
                "major": item.major,
                "intent": item.intent,
                "handler": item.handler,
                "tools": tools,
                "observe": observe,
            }
        )
    return tasks


def turn_result_to_payload(
    *,
    prep: CompanionPreprocess,
    query: str,
    items: list[CompanionIntentResult],
    backend: str,
) -> dict[str, Any]:
    observe = format_preprocess_for_agent(prep)
    tasks = build_intent_tasks(items, observe=observe)
    bound: list[str] = []
    for task in tasks:
        bound.extend(str(getattr(tool, "name", "") or "") for tool in task.get("tools") or [])
    primary = items[0] if items else None
    return {
        "ok": True,
        "raw_text": prep.raw_text,
        "text": query,
        "backend": (backend or "llm").strip().lower(),
        "preprocess": prep.to_dict(),
        "observe": observe,
        "items": items,
        "intent_tasks": [intent_result_to_dict(item) for item in items],
        "tasks": tasks,
        "bound_tools": bound,
        "major": primary.major if primary else "",
        "intent": primary.intent if primary else "",
        "handler": primary.handler if primary else "",
        "source": primary.source if primary else "",
        "confidence": float(primary.confidence) if primary else 0.0,
        "reason": primary.reason if primary else "",
        "slots": list(primary.slots or ()) if primary else [],
        "recommend_action": primary.recommend_action if primary else "",
        "major_scores": dict(primary.major_scores or {}) if primary else {},
    }


def attach_intent_to_preprocess(
    prep: CompanionPreprocess,
    model: Any = None,
    *,
    ctx: CompanionTurnContext | None = None,
    backend: str = "llm",
    apply_file_route: bool = True,
) -> dict[str, Any]:
    """陪聊 _attach_intent：在已有 preprocess 上挂意图与抽槽。"""
    query = prep.resolved_text or prep.corrected_text or prep.raw_text
    if not (query or "").strip():
        return {"ok": False, "reason": "empty"}
    context = ctx or CompanionTurnContext()
    t_intent = time.perf_counter()
    items = resolve_companion_intent_items(
        query.strip(),
        model,
        recent_dialogue=(context.recent_dialogue or "").strip(),
        backend=backend,
    )
    intent_ms = int((time.perf_counter() - t_intent) * 1000)
    if apply_file_route:
        items = apply_file_route_to_intents(
            query,
            items,
            last_file=context.last_file,
        )
    finished = finish_preprocess_after_intent(prep, items)
    finished.stage_ms = {**(finished.stage_ms or {}), "intent_ms": intent_ms}
    return turn_result_to_payload(
        prep=finished,
        query=query.strip(),
        items=items,
        backend=backend,
    )


def resolve_companion_turn(
    text: str,
    model: Any = None,
    *,
    ctx: CompanionTurnContext | None = None,
    backend: str = "llm",
    apply_file_route: bool = True,
) -> dict[str, Any]:
    """与陪聊 _begin_turn + _attach_intent 一致（不含 Agent 调用与 session 持久化）。"""
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty"}
    context = ctx or CompanionTurnContext()
    prep = run_chat_preprocess(
        raw,
        user_id=context.user_id,
        last_preprocess=context.last_preprocess,
        last_user_text=context.last_user_text,
        slot_memory=context.slot_memory,
        prior_user_texts=context.prior_user_texts,
        defer_intent=True,
    )
    return attach_intent_to_preprocess(
        prep,
        model,
        ctx=context,
        backend=backend,
        apply_file_route=apply_file_route,
    )
