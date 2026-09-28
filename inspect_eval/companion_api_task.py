"""Inspect AI：HTTP 调 C 端陪聊 API（需先 run.py :8002）。

    inspect eval inspect_eval/companion_api_task.py:companion_smoke --model mockllm/model
    inspect view
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

__all__ = ["companion_smoke", "companion_intent_samples", "companion_multi_intent_case"]

_BASE = (os.getenv("B2C_INSPECT_API_BASE") or "http://127.0.0.1:8002").rstrip("/")
_USER = (os.getenv("B2C_INSPECT_USER_ID") or "5cc279c7-f757-463e-9552-44c225a4cc70").strip()
_TIMEOUT = float(os.getenv("B2C_INSPECT_HTTP_TIMEOUT") or "180")


async def _companion_chat(text: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        sess = await client.post(
            f"{_BASE}/api/sessions",
            json={"user_id": _USER, "force_new": True},
        )
        sess.raise_for_status()
        body = sess.json()
        sid = str(body.get("session_id") or "")
        if not sid:
            raise RuntimeError(f"no session_id: {body}")
        chat = await client.post(
            f"{_BASE}/api/sessions/{sid}/chat",
            json={"text": text},
        )
        chat.raise_for_status()
        return chat.json()


@solver
def companion_api_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        query = (state.input_text or "").strip()
        try:
            payload = await _companion_chat(query)
        except Exception as exc:
            state.output.completion = f"[B2C API error] {exc}"
            state.metadata["b2c_error"] = str(exc)
            return state
        reply = str(payload.get("reply") or "").strip()
        state.output.completion = reply or "(empty reply)"
        rnd = payload.get("round") if isinstance(payload.get("round"), dict) else {}
        lg = rnd.get("langgraph") if isinstance(rnd.get("langgraph"), dict) else {}
        state.metadata["b2c_round"] = {
            "bound_tools": rnd.get("bound_tools") or [],
            "tool_runs": len(lg.get("tool_runs") or []),
            "pattern": lg.get("pattern"),
            "intent_tasks": rnd.get("intent_tasks") or [],
        }
        prep = rnd.get("preprocess") if isinstance(rnd.get("preprocess"), dict) else {}
        if prep:
            state.metadata["b2c_preprocess"] = {
                "intent_major": prep.get("intent_major"),
                "intent": prep.get("intent"),
                "handler": prep.get("intent_handler"),
            }
        state.metadata["b2c_reply_len"] = len(reply)
        return state

    return solve


@scorer(metrics=[accuracy()])
def companion_reply_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        text = (state.output.completion or "").strip()
        if text.startswith("[B2C API error]"):
            return Score(value=0.0, answer=text, explanation="API 调用失败")
        want = (target.text or "").strip()
        if not text or text == "(empty reply)":
            return Score(value=0.0, answer=text, explanation="空回复")
        if not want:
            return Score(value=1.0, answer=text[:500])
        ok = want in text
        return Score(
            value=1.0 if ok else 0.0,
            answer=text[:800],
            explanation=f"期望包含：{want}" if not ok else None,
        )

    return score


@task
def companion_smoke() -> Task:
    return Task(
        dataset=[
            Sample(input="你好", target=""),
            Sample(input="谢谢", target=""),
        ],
        solver=[companion_api_solver()],
        scorer=companion_reply_scorer(),
    )


@task
def companion_intent_samples() -> Task:
    return Task(
        dataset=[
            Sample(
                input="头部常见有哪些病",
                target="",
                metadata={"note": "公共知识 / kg"},
            ),
            Sample(
                input="我这两天心情不太好",
                target="",
                metadata={"note": "近况陈述"},
            ),
        ],
        solver=[companion_api_solver()],
        scorer=companion_reply_scorer(),
    )


_MULTI_INTENT_CASE = (
    "最近老觉得乏力、口干，想同时了解乏力常见和哪些病有关、"
    "口干生活上要注意什么、以及要不要先查个血常规是查什么的。"
)


@task
def companion_multi_intent_case() -> Task:
    """多意图一句：知识 + 生活注意 + 检查含义（走 8002 全链路）。"""
    return Task(
        dataset=[
            Sample(
                input=_MULTI_INTENT_CASE,
                target="",
                metadata={"note": "多意图并行 + handoff 样例"},
            ),
        ],
        solver=[companion_api_solver()],
        scorer=companion_reply_scorer(),
    )
