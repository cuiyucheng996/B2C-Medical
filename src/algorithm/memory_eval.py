"""个人记忆评测：细意图分类 + 工具绑定 + 直调 memory 工具。"""

from __future__ import annotations

import json
import time
from typing import Any

from algorithm.intent_catalog import INTENT_ROWS
from algorithm.companion_turn import resolve_companion_turn
from algorithm.intent_classify import CompanionIntentResult
from tools.companion_memory import companion_user_id
from tools.companion_tools import companion_turn, tools_for_intent
from tools.setup import INTENT_BIND
from tools.toolkits import MEMORY_TOOLKIT

__all__ = [
    "MEM_INTENT_SAMPLES",
    "MEM_BIND_CASES",
    "list_memory_handlers",
    "list_memory_probe_tools",
    "run_memory_intent_gold",
    "run_memory_bind_gold",
    "run_memory_intent_one",
    "run_memory_tool_probe",
    "run_memory_qa",
]

_MAJOR = "个人记忆问答"
_MEM_TOOL_MAP = {str(tool.name): tool for tool in MEMORY_TOOLKIT.get_tools()}

MEM_INTENT_SAMPLES: list[dict[str, Any]] = [
    {
        "id": "stated_part",
        "text": "我说过哪里不舒服",
        "major": "个人记忆问答",
        "handler": "mem_stated_consult",
    },
    {
        "id": "stated_disease",
        "text": "我之前跟你提过什么病",
        "major": "个人记忆问答",
        "handler": "mem_stated_consult",
    },
    {
        "id": "life_mood",
        "text": "我上次跟你聊的心情还记得吗",
        "major": "个人记忆问答",
        "handler": "mem_companion_fact",
    },
    {
        "id": "life_event",
        "text": "我跟你说过出差的事吗",
        "major": "个人记忆问答",
        "handler": "mem_companion_fact",
    },
    {
        "id": "session_recall",
        "text": "我刚才说什么了",
        "major": "个人记忆问答",
        "handler": "mem_session_recall",
    },
    {
        "id": "mem_generic",
        "text": "你还记得我以前说过什么",
        "major": "个人记忆问答",
        "handler": "",
    },
]

MEM_BIND_CASES: list[dict[str, Any]] = [
    {
        "id": "bind_mem_stated",
        "major": "个人记忆问答",
        "handler": "mem_stated_consult",
        "tools": ["lookup_stated_consult"],
    },
    {
        "id": "bind_mem_life",
        "major": "个人记忆问答",
        "handler": "mem_companion_fact",
        "tools": ["lookup_life_facts"],
    },
    {
        "id": "bind_mem_session_recall",
        "major": "个人记忆问答",
        "handler": "mem_session_recall",
        "tools": [],
    },
    {
        "id": "bind_mem_default",
        "major": "个人记忆问答",
        "handler": "",
        "tools": ["lookup_stated_consult", "lookup_life_facts"],
    },
]


def _tool_names(major: str, handler: str) -> list[str]:
    return [str(getattr(item, "name", "") or "") for item in tools_for_intent(major=major, handler=handler)]


def list_memory_handlers() -> list[dict[str, str]]:
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for item in INTENT_ROWS:
        if str(item.get("major") or "") != _MAJOR:
            continue
        handler = str(item.get("handler") or "").strip()
        if not handler or handler in seen:
            continue
        seen.add(handler)
        rows.append({"handler": handler, "intent": str(item.get("intent") or "")})
    rows.append({"handler": "默认", "intent": "大类默认（双工具）"})
    return rows


def list_memory_probe_tools() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool in MEMORY_TOOLKIT.get_tools():
        desc = str(getattr(tool, "description", "") or "").strip().split("\n")[0]
        rows.append(
            {
                "name": str(tool.name),
                "title": str(tool.name),
                "summary": desc,
                "invokable": True,
                "source": "local",
            }
        )
    rows.append(
        {
            "name": "graphiti_mcp",
            "title": "MCP Graphiti 记忆",
            "summary": "search_memory_facts / search_nodes（陪聊 Agent 经 MCP 调用，评测台不直调）",
            "invokable": False,
            "source": "mcp",
            "allow_tools": ["search_memory_facts", "search_nodes"],
        }
    )
    return rows


def _primary_from_text(text: str) -> CompanionIntentResult | None:
    turn = resolve_companion_turn(text, backend="llm", apply_file_route=False)
    if not turn.get("ok"):
        return None
    items = turn.get("items") or []
    return items[0] if items else None


def _score_intent(case: dict[str, Any]) -> dict[str, Any]:
    hit = _primary_from_text(str(case["text"]))
    if hit is None:
        return {
            "id": case["id"],
            "kind": "intent",
            "ok": False,
            "text": case["text"],
            "fail": ["turn"],
        }
    tools = _tool_names(hit.major, hit.handler)
    want_major = str(case.get("major") or "")
    want_handler = str(case.get("handler") or "")
    ok_major = (not want_major) or hit.major == want_major
    ok_handler = True
    if want_handler:
        ok_handler = hit.handler == want_handler
    ok = ok_major and ok_handler
    return {
        "id": case["id"],
        "kind": "intent",
        "ok": ok,
        "text": case["text"],
        "expect_major": want_major,
        "expect_handler": want_handler or "任意/默认",
        "got_major": hit.major,
        "got_handler": hit.handler or "-",
        "got_intent": hit.intent,
        "got_source": hit.source,
        "got_tools": tools,
        "fail": [] if ok else [
            *([] if ok_major else ["major"]),
            *([] if ok_handler else ["handler"]),
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
        "got_tools": got,
        "expect_tools": want,
        "fail": [] if ok else ["tools"],
    }


def run_memory_intent_gold() -> dict[str, Any]:
    rows = [_score_intent(c) for c in MEM_INTENT_SAMPLES]
    passed = sum(1 for r in rows if r.get("ok"))
    return {
        "ok": passed == len(rows),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }


def run_memory_bind_gold() -> dict[str, Any]:
    rows = [_score_bind(c) for c in MEM_BIND_CASES]
    passed = sum(1 for r in rows if r.get("ok"))
    return {
        "ok": passed == len(rows),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
        "bind_catalog": [
            {"major": m, "handler": h or "默认", "tools": list(n)}
            for (m, h), n in INTENT_BIND.items()
            if m == _MAJOR
        ],
    }


def run_memory_intent_one(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"ok": False, "reason": "empty"}
    hit = _primary_from_text(raw)
    if hit is None:
        return {"ok": False, "reason": "turn_failed"}
    tools = _tool_names(hit.major, hit.handler)
    bind_key = (_MAJOR, hit.handler or "") if hit.major == _MAJOR else (hit.major, hit.handler or "")
    expect_tools = list(INTENT_BIND.get(bind_key) or INTENT_BIND.get((hit.major, "")) or ())
    return {
        "ok": True,
        "text": raw,
        "major": hit.major,
        "intent": hit.intent,
        "handler": hit.handler,
        "source": hit.source,
        "confidence": hit.confidence,
        "tools": tools,
        "expect_tools_if_routed": expect_tools if hit.major == _MAJOR else [],
        "is_memory_major": hit.major == _MAJOR,
    }


def run_memory_tool_probe(
    *,
    user_id: str,
    query: str,
    tools: list[str],
) -> dict[str, Any]:
    uid = (user_id or "").strip()
    raw = (query or "").strip()
    allowed = set(_MEM_TOOL_MAP)
    picked = [name for name in (tools or []) if name in allowed]
    if not uid:
        return {"ok": False, "reason": "no_user_id", "available_tools": sorted(allowed)}
    if not picked:
        return {"ok": False, "reason": "no_tools", "available_tools": sorted(allowed)}
    t0 = time.perf_counter()
    uid_token = companion_user_id.set(uid)
    base_turn = {"user_id": uid, "query": raw or "记忆检索"}
    tool_rows: list[dict[str, Any]] = []
    try:
        for name in picked:
            tool = _MEM_TOOL_MAP.get(name)
            if tool is None:
                tool_rows.append({"tool": name, "ok": False, "error": "unknown_tool", "elapsed_ms": 0})
                continue
            turn_token = companion_turn.set(dict(base_turn))
            t_tool = time.perf_counter()
            try:
                if name == "lookup_stated_consult":
                    out = tool.invoke({})
                else:
                    out = tool.invoke({"query": raw or "近况"})
            except Exception as exc:
                elapsed = int((time.perf_counter() - t_tool) * 1000)
                tool_rows.append(
                    {"tool": name, "ok": False, "error": str(exc), "elapsed_ms": elapsed}
                )
                companion_turn.reset(turn_token)
                continue
            elapsed = int((time.perf_counter() - t_tool) * 1000)
            companion_turn.reset(turn_token)
            parsed: dict[str, Any]
            if isinstance(out, str):
                try:
                    parsed = json.loads(out)
                except json.JSONDecodeError:
                    parsed = {"ok": True, "summary": out, "data": {}}
            elif isinstance(out, dict):
                parsed = out
            else:
                parsed = {"ok": True, "summary": str(out), "data": {}}
            tool_rows.append(
                {
                    "tool": name,
                    "ok": bool(parsed.get("ok", True)),
                    "usable": parsed.get("usable"),
                    "summary": parsed.get("summary") or "",
                    "data": parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed,
                    "elapsed_ms": elapsed,
                }
            )
    finally:
        companion_user_id.reset(uid_token)
    total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "ok": True,
        "user_id": uid,
        "query": raw,
        "tool_results": tool_rows,
        "timing": {"total_ms": total_ms},
        "available_tools": sorted(allowed),
    }


def run_memory_qa(*, user_id: str, query: str) -> dict[str, Any]:
    """先分类，再对该用户 invoke 本轮绑定工具。"""
    uid = (user_id or "").strip()
    if not uid:
        return {"ok": False, "reason": "no_user_id"}
    intent = run_memory_intent_one(query)
    if not intent.get("ok"):
        return intent
    intent["user_id"] = uid
    tools = list(intent.get("tools") or [])
    if not intent.get("is_memory_major"):
        intent["tool_results"] = []
        intent["tool_skip"] = "not_memory_major"
        return intent
    if not tools:
        intent["tool_results"] = []
        intent["tool_skip"] = "no_tools_bound"
        return intent
    probe = run_memory_tool_probe(user_id=uid, query=query, tools=tools)
    intent["tool_results"] = list(probe.get("tool_results") or [])
    intent["timing"] = probe.get("timing") or {}
    return intent
