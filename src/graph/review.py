"""评审 Agent：读执行草稿，结构化 pass/fail（与 Think 信封审核分开）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from algorithm.tool_think import _think_llm
from prompt import load_prompt

logger = logging.getLogger(__name__)

__all__ = ["review_companion_draft"]

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _parse_review(text: str) -> dict[str, Any]:
    raw = _FENCE.sub("", (text or "").strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {"pass": True, "reason": "评审结果无法解析，放行", "hint": ""}
    try:
        row = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"pass": True, "reason": "评审结果无法解析，放行", "hint": ""}
    if not isinstance(row, dict):
        return {"pass": True, "reason": "评审结果无法解析，放行", "hint": ""}
    passed = bool(row.get("pass"))
    return {
        "pass": passed,
        "reason": str(row.get("reason") or "").strip()[:80],
        "hint": "" if passed else str(row.get("hint") or "").strip()[:120],
    }


def review_companion_draft(
    *,
    query: str,
    major: str,
    intent: str,
    handler: str,
    draft: str,
) -> dict[str, Any]:
    """无评审模型时放行，避免把线上陪聊卡死。"""
    text = (draft or "").strip()
    if not text:
        return {"pass": False, "reason": "执行草稿为空", "hint": "按细意图与观察调用工具后给出完整回复"}
    llm = _think_llm()
    if llm is None:
        return {"pass": True, "reason": "未配置评审模型，放行", "hint": ""}
    human = (
        f"细意图：{major} / {intent}（{handler}）\n"
        f"用户问题：{query or '（空）'}\n"
        f"执行草稿：\n{text[:3000]}"
    )
    try:
        from algorithm.llm_trace import STAGE_REVIEW, record_llm_call, serialize_langchain_messages

        msgs = [SystemMessage(content=load_prompt("companion_review")), HumanMessage(content=human)]
        resp = llm.invoke(msgs)
        content = str(getattr(resp, "content", "") or "")
        record_llm_call(
            STAGE_REVIEW,
            serialize_langchain_messages(msgs),
            response=content,
            label=f"{major} / {intent} · {handler}",
        )
        return _parse_review(content)
    except Exception:
        logger.warning("companion review llm failed", exc_info=True)
        return {"pass": True, "reason": "评审调用失败，放行", "hint": ""}
