"""Inspect AI：陪聊 Agent 离线评测任务（意图金标 / 工具绑定 / 完整图冒烟）。"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver

__all__ = [
    "TASK_REGISTRY",
    "companion_intent_gold",
    "companion_tool_bind",
    "companion_graph_smoke",
]


def _gold_target(case: dict[str, Any]) -> str:
    return json.dumps(
        {
            "major": str(case.get("major") or ""),
            "handler": str(case.get("handler") or ""),
            "tools_empty": bool(case.get("tools_empty")),
        },
        ensure_ascii=False,
    )


def _score_gold_payload(got_row: dict[str, Any], want: dict[str, Any]) -> tuple[bool, str]:
    if not got_row.get("ok"):
        return False, got_row.get("reason") or "classify_failed"
    want_major = str(want.get("major") or "")
    want_handler = str(want.get("handler") or "")
    got_major = str(got_row.get("major") or "")
    got_handler = str(got_row.get("handler") or "")
    tools = list(got_row.get("tools") or [])
    ok_major = (not want_major) or got_major == want_major
    ok_handler = (not want_handler) or got_handler == want_handler
    ok_empty = True
    if want.get("tools_empty"):
        ok_empty = tools == []
    ok = ok_major and ok_handler and ok_empty
    parts: list[str] = []
    if not ok_major:
        parts.append(f"major {got_major}!={want_major}")
    if not ok_handler:
        parts.append(f"handler {got_handler}!={want_handler}")
    if not ok_empty:
        parts.append(f"tools={tools}")
    return ok, "; ".join(parts) if parts else "match"


@solver
def companion_classify_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from algorithm.eval_suite import classify_one

        row = classify_one(state.input_text)
        state.output.completion = json.dumps(row, ensure_ascii=False)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def companion_gold_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            got_row = json.loads(state.output.completion or "{}")
            want = json.loads(target.text or "{}")
        except json.JSONDecodeError:
            return Score(value="I", explanation="invalid_json")
        ok, detail = _score_gold_payload(got_row, want)
        return Score(value="C" if ok else "I", answer=got_row.get("major"), explanation=detail)

    return score


@task
def companion_intent_gold() -> Task:
    from algorithm.eval_suite import GOLD_CASES

    samples = [
        Sample(input=str(c["text"]), target=_gold_target(c), id=str(c["id"]))
        for c in GOLD_CASES
    ]
    return Task(
        dataset=samples,
        solver=companion_classify_solver(),
        scorer=companion_gold_scorer(),
        name="companion_intent_gold",
    )


@solver
def companion_bind_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from tools.setup import INTENT_BIND

        major = (state.input_text or "").strip()
        meta = state.metadata if isinstance(state.metadata, dict) else {}
        handler = str(meta.get("handler") or "")
        key = (major, handler)
        if key not in INTENT_BIND:
            key = (major, "")
        names = list(INTENT_BIND.get(key, ()))
        state.output.completion = json.dumps({"tools": names}, ensure_ascii=False)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def companion_bind_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            got = json.loads(state.output.completion or "{}")
            want_tools = json.loads(target.text or "[]")
        except json.JSONDecodeError:
            return Score(value="I", explanation="invalid_json")
        got_tools = list(got.get("tools") or [])
        ok = got_tools == want_tools
        return Score(
            value="C" if ok else "I",
            answer=",".join(got_tools),
            explanation="match" if ok else f"got={got_tools} want={want_tools}",
        )

    return score


@task
def companion_tool_bind() -> Task:
    from algorithm.eval_suite import BIND_CASES

    samples = [
        Sample(
            input=str(c["major"]),
            target=json.dumps(list(c.get("tools") or []), ensure_ascii=False),
            id=str(c["id"]),
            metadata={"handler": str(c.get("handler") or "")},
        )
        for c in BIND_CASES
    ]
    return Task(
        dataset=samples,
        solver=companion_bind_solver(),
        scorer=companion_bind_scorer(),
        name="companion_tool_bind",
    )


_GRAPH_SMOKE: list[Sample] = [
    Sample(input="你好", target=json.dumps({"min_reply_len": 1}), id="smoke_hi"),
    Sample(input="头部常见有哪些病", target=json.dumps({"min_reply_len": 8}), id="smoke_kg"),
    Sample(input="胃炎有哪些症状", target=json.dumps({"min_reply_len": 8}), id="smoke_symptom"),
]


@solver
def companion_graph_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from algorithm.graph_eval import run_graph_probe

        row = await run_graph_probe(state.input_text, mode="full")
        state.output.completion = json.dumps(row, ensure_ascii=False)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def companion_graph_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            row = json.loads(state.output.completion or "{}")
            want = json.loads(target.text or "{}")
        except json.JSONDecodeError:
            return Score(value="I", explanation="invalid_json")
        min_len = int(want.get("min_reply_len") or 1)
        reply = str(row.get("reply") or "")
        ok = bool(row.get("ok")) and len(reply) >= min_len
        detail = "ok" if ok else f"ok={row.get('ok')} len={len(reply)} reason={row.get('reason') or row.get('error')}"
        return Score(value="C" if ok else "I", answer=reply[:120], explanation=detail)

    return score


@task
def companion_graph_smoke() -> Task:
    return Task(
        dataset=_GRAPH_SMOKE,
        solver=companion_graph_solver(),
        scorer=companion_graph_scorer(),
        name="companion_graph_smoke",
    )


TASK_REGISTRY: dict[str, dict[str, Any]] = {
    "intent_gold": {
        "title": "意图金标",
        "description": "与评测台「其它」金标同源：preprocess → 大类/细意图/handler，对照 GOLD_CASES。",
        "factory": companion_intent_gold,
        "default_limit": None,
    },
    "tool_bind": {
        "title": "工具绑定",
        "description": "按 major+handler 核对 INTENT_BIND 暴露的工具名列表。",
        "factory": companion_tool_bind,
        "default_limit": None,
    },
    "graph_smoke": {
        "title": "完整图冒烟",
        "description": "跑 LangGraph 完整陪聊图，检查有回复且长度达标（需聊天模型）。",
        "factory": companion_graph_smoke,
        "default_limit": 3,
    },
}
