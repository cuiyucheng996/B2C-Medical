"""离线评测：意图金标 + 工具绑定。与陪聊同一套进线解析。"""

from __future__ import annotations

from typing import Any

from algorithm.companion_turn import resolve_companion_turn
from tools.companion_tools import tools_for_intent
from tools.setup import INTENT_BIND

__all__ = ["GOLD_CASES", "BIND_CASES", "run_eval", "probe_env", "classify_one"]

GOLD_CASES: list[dict[str, Any]] = [
    {"id": "chat_hi", "text": "你好", "major": "闲聊", "tools_empty": True},
    {"id": "kg_part", "text": "头部常见有哪些病", "major": "公共知识问答"},
    {"id": "kg_symptom", "text": "胃炎有哪些症状", "major": "公共知识问答"},
    {"id": "kg_exam", "text": "血常规这个检查是查什么的", "major": "公共知识问答"},
    {"id": "task_remind", "text": "提醒我复查", "major": "具体任务", "handler": "task_remind"},
    {"id": "task_med", "text": "提醒我吃药", "major": "具体任务", "handler": "task_medication"},
    {"id": "task_appt", "text": "帮我挂号", "major": "具体任务", "handler": "task_appointment"},
    {"id": "mem", "text": "我说过哪里不舒服", "major": "个人记忆问答"},
    {"id": "report", "text": "帮我看看这份化验单", "major": "报告指标解读"},
    {"id": "risk", "text": "这个药一天几次", "major": "风险拦截"},
    {"id": "life", "text": "我头疼好几天了", "major": "近况陈述"},
]

BIND_CASES: list[dict[str, Any]] = [
    {"id": "bind_chat", "major": "闲聊", "handler": "chat_hello", "tools": []},
    {"id": "bind_graph", "major": "公共知识问答", "handler": "kg_part_diseases", "tools": ["query_consult_graph"]},
    {"id": "bind_mem", "major": "个人记忆问答", "handler": "mem_stated_consult", "tools": ["lookup_stated_consult"]},
    {"id": "bind_report", "major": "报告指标解读", "handler": "report_upload", "tools": ["explain_uploaded_report", "extract_report_tables"]},
    {"id": "bind_task", "major": "具体任务", "handler": "task_remind", "tools": ["note_task_remind"]},
    {"id": "bind_risk", "major": "风险拦截", "handler": "", "tools": ["safety_policy_hint"]},
]


def _tool_names(major: str, handler: str) -> list[str]:
    return [str(getattr(t, "name", "") or "") for t in tools_for_intent(major=major, handler=handler)]


def _primary_hit(text: str) -> Any:
    turn = resolve_companion_turn(text, backend="llm", apply_file_route=False)
    if not turn.get("ok"):
        return None
    items = turn.get("items") or []
    return items[0] if items else None


def _score_gold(case: dict[str, Any]) -> dict[str, Any]:
    hit = _primary_hit(str(case["text"]))
    if hit is None:
        return {
            "id": case["id"],
            "kind": "gold",
            "ok": False,
            "text": case["text"],
            "fail": ["turn"],
        }
    tools = _tool_names(hit.major, hit.handler)
    want_major = str(case.get("major") or "")
    want_handler = str(case.get("handler") or "")
    ok_major = (not want_major) or hit.major == want_major
    ok_handler = (not want_handler) or hit.handler == want_handler
    ok_empty = True
    if case.get("tools_empty"):
        ok_empty = tools == []
    ok = ok_major and ok_handler and ok_empty
    return {
        "id": case["id"],
        "kind": "gold",
        "ok": ok,
        "text": case["text"],
        "expect_major": want_major,
        "expect_handler": want_handler or "-",
        "got_major": hit.major,
        "got_handler": hit.handler or "-",
        "got_intent": hit.intent,
        "got_source": hit.source,
        "got_tools": tools,
        "fail": [] if ok else [
            *([] if ok_major else ["major"]),
            *([] if ok_handler else ["handler"]),
            *([] if ok_empty else ["tools_not_empty"]),
        ],
    }


def _score_bind(case: dict[str, Any]) -> dict[str, Any]:
    got = _tool_names(str(case["major"]), str(case.get("handler") or ""))
    want = list(case.get("tools") or [])
    ok = got == want
    return {
        "id": case["id"],
        "kind": "bind",
        "ok": ok,
        "text": f"{case['major']} / {case.get('handler') or '默认'}",
        "expect_major": case["major"],
        "expect_handler": case.get("handler") or "默认",
        "got_major": case["major"],
        "got_handler": case.get("handler") or "默认",
        "got_intent": "",
        "got_source": "INTENT_BIND",
        "got_tools": got,
        "fail": [] if ok else ["tools"],
        "expect_tools": want,
    }


def classify_one(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty"}
    turn = resolve_companion_turn(raw, backend="llm", apply_file_route=False)
    if not turn.get("ok"):
        return turn
    hit = (turn.get("items") or [None])[0]
    if hit is None:
        return {"ok": False, "reason": "no_intent"}
    return {
        "ok": True,
        "text": raw,
        "query_after_preprocess": turn.get("text") or raw,
        "preprocess": turn.get("preprocess") or {},
        "major": hit.major,
        "intent": hit.intent,
        "handler": hit.handler,
        "source": hit.source,
        "confidence": hit.confidence,
        "tools": list(turn.get("bound_tools") or []),
    }


def run_eval(*, extra_texts: list[str] | None = None) -> dict[str, Any]:
    gold = [_score_gold(c) for c in GOLD_CASES]
    binds = [_score_bind(c) for c in BIND_CASES]
    extras = []
    for text in extra_texts or []:
        row = classify_one(text)
        extras.append(row)
    rows = gold + binds
    passed = sum(1 for r in rows if r.get("ok"))
    return {
        "ok": passed == len(rows),
        "passed": passed,
        "total": len(rows),
        "gold": gold,
        "binds": binds,
        "extras": extras,
        "bind_catalog": [
            {"major": m, "handler": h or "默认", "tools": list(n)}
            for (m, h), n in INTENT_BIND.items()
        ],
    }


def probe_env() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("api", True, "eval 进程内")
    try:
        from algorithm.user_accounts import list_users

        users = list_users(limit=20)
        add("neo4j_userinfo", True, f"{len(users)} 条 UserInfo")
    except Exception as exc:
        add("neo4j_userinfo", False, str(exc))
    try:
        from algorithm.voice_client import voice_status

        st = voice_status()
        add("voice", bool(st.get("asr") and st.get("tts")), str(st)[:240])
    except Exception as exc:
        add("voice", False, str(exc))
    try:
        from algorithm.ragflow_kb import kb_status

        kb = kb_status()
        add("ragflow", bool(kb.get("reachable")), str(kb.get("error") or kb.get("web_url") or kb)[:240])
    except Exception as exc:
        add("ragflow", False, str(exc))
    return {"checks": checks, "ok": all(c["ok"] for c in checks if c["name"] != "ragflow")}
