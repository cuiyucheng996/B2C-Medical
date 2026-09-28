"""评测：与陪聊相同的 preprocess → 意图 → 抽槽。"""

from __future__ import annotations

from typing import Any

from algorithm.companion_turn import CompanionTurnContext, resolve_companion_turn
from algorithm.eval_suite import GOLD_CASES
from algorithm.intent_catalog import INTENT_ROWS
from algorithm.intent_classify import (
    intent_result_to_dict,
    major_bert_available,
    major_score_threshold,
    major_scores_to_rows,
)
from tools.companion_tools import tools_for_intent

__all__ = [
    "INTENT_TRY_SAMPLES",
    "intent_catalog_rows",
    "run_intent_gold",
    "run_intent_one",
]


INTENT_TRY_SAMPLES: list[dict[str, str]] = [
    {"text": item["text"], "note": item.get("id") or ""}
    for item in GOLD_CASES
] + [
    {"text": "我上次跟你聊的心情还记得吗", "note": "个人记忆·心情"},
    {"text": "你还记得我以前说过什么", "note": "个人记忆·泛问"},
]


def intent_catalog_rows() -> list[dict[str, Any]]:
    return [
        {
            "major": str(r.get("major") or ""),
            "intent": str(r.get("intent") or ""),
            "handler": str(r.get("handler") or ""),
            "slots": list(r.get("slots") or ()),
        }
        for r in INTENT_ROWS
    ]


def run_intent_one(
    text: str,
    *,
    backend: str = "llm",
    recent_dialogue: str = "",
    slot_memory: dict[str, list[str]] | None = None,
    last_user_text: str = "",
) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty"}
    mode = (backend or "llm").strip().lower()
    if mode not in {"llm", "bert"}:
        return {"ok": False, "reason": f"unknown backend: {backend}"}
    if mode == "bert" and not major_bert_available():
        return {
            "ok": False,
            "reason": "BERT 大意图模型不可用，请放置 checkpoint/intent_major/best_model",
        }
    turn = resolve_companion_turn(
        raw,
        ctx=CompanionTurnContext(
            recent_dialogue=(recent_dialogue or "").strip(),
            slot_memory=slot_memory,
            last_user_text=(last_user_text or "").strip(),
        ),
        backend=mode,
        apply_file_route=False,
    )
    if not turn.get("ok"):
        return turn
    items = list(turn.get("items") or [])
    hit = items[0] if items else None
    if hit is None:
        return {"ok": False, "reason": "no_intent"}
    return {
        "ok": True,
        "text": raw,
        "query_after_preprocess": turn.get("text") or raw,
        "preprocess": turn.get("preprocess") or {},
        "backend": mode,
        "recent_dialogue_used": bool((recent_dialogue or "").strip()),
        "major": hit.major,
        "intent": hit.intent,
        "handler": hit.handler or "",
        "source": hit.source,
        "confidence": hit.confidence,
        "reason": hit.reason,
        "slots": list(hit.slots or ()),
        "recommend_action": hit.recommend_action,
        "intents": [intent_result_to_dict(item) for item in items],
        "major_scores": major_scores_to_rows(hit.major_scores),
        "major_score_threshold": major_score_threshold(),
        "executable_intents": [intent_result_to_dict(item) for item in items],
        "bound_tools": list(turn.get("bound_tools") or []),
    }


def run_intent_gold() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in GOLD_CASES:
        turn = resolve_companion_turn(str(case["text"]), backend="llm", apply_file_route=False)
        if not turn.get("ok"):
            rows.append({"id": case.get("id"), "ok": False, "text": case["text"], "fail": ["turn"]})
            continue
        hit = (turn.get("items") or [None])[0]
        if hit is None:
            rows.append({"id": case.get("id"), "ok": False, "text": case["text"], "fail": ["no_intent"]})
            continue
        want_major = str(case.get("major") or "")
        want_handler = str(case.get("handler") or "")
        ok_major = (not want_major) or hit.major == want_major
        ok_handler = (not want_handler) or hit.handler == want_handler
        ok_empty = True
        if case.get("tools_empty"):
            bound = [
                str(getattr(t, "name", "") or "")
                for t in tools_for_intent(major=hit.major, handler=hit.handler)
            ]
            ok_empty = bound == []
        ok = ok_major and ok_handler and ok_empty
        rows.append(
            {
                "id": case.get("id"),
                "ok": ok,
                "text": case["text"],
                "expect_major": want_major,
                "expect_handler": want_handler or "-",
                "expect_intent": "",
                "got_major": hit.major,
                "got_intent": hit.intent,
                "got_handler": hit.handler or "-",
                "got_source": hit.source,
                "fail": [] if ok else [
                    *([] if ok_major else ["major"]),
                    *([] if ok_handler else ["handler"]),
                    *([] if ok_empty else ["should_be_chat"]),
                ],
            }
        )
    passed = sum(1 for r in rows if r.get("ok"))
    return {
        "ok": passed == len(rows),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
        "catalog": intent_catalog_rows(),
    }
