"""评审 Agent：读执行草稿，结构化 pass/fail（与 Think 信封审核分开）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from algorithm.structured_output import invoke_structured_llm
from algorithm.tool_think import _think_llm
from prompt import load_prompt
from prompt.structured_block import compose_structured_system_prompt
from schema.llm_outputs import CompanionReviewOutput

logger = logging.getLogger(__name__)

__all__ = [
    "companion_draft_review_mode",
    "companion_should_review_draft",
    "review_companion_draft",
]

_DraftReviewMode = Literal["off", "all", "public"]
_PUBLIC_KNOWLEDGE_MAJOR = "公共知识问答"

_REVIEW_EXAMPLES = [
    {"pass": True, "reason": "对准化验解读", "hint": ""},
    {"pass": False, "reason": "编造了未给出的指标", "hint": "只使用工具 data 里出现的数值"},
]


def _review_system_prompt() -> str:
    return compose_structured_system_prompt(
        load_prompt("companion_review"),
        CompanionReviewOutput,
        examples=_REVIEW_EXAMPLES,
    )


def _row_to_dict(row: CompanionReviewOutput) -> dict[str, Any]:
    passed = bool(row.pass_)
    return {
        "pass": passed,
        "reason": str(row.reason or "").strip()[:80],
        "hint": "" if passed else str(row.hint or "").strip()[:120],
    }


def _review_fallback() -> CompanionReviewOutput:
    return CompanionReviewOutput(pass_=True, reason="评审结构化失败，放行", hint="")


def _normalize_draft_review_mode(raw: str) -> _DraftReviewMode:
    key = (raw or "off").strip().lower()
    if key in {"0", "false", "no", "off", "disabled", ""}:
        return "off"
    if key in {"1", "true", "yes", "on", "all"}:
        return "all"
    if key in {"public", "public_knowledge", "kg", "knowledge"}:
        return "public"
    return "off"


def _draft_review_mode_from_yaml() -> str:
    try:
        from src.configs import get_yaml_config

        block = (get_yaml_config() or {}).get("companion") or {}
        return str(block.get("draft_review") or "").strip()
    except Exception:
        return ""


def companion_draft_review_mode() -> _DraftReviewMode:
    """off=不调评审 LLM；all=每路子图都审；public=仅公共知识问答大类。"""
    env = (os.getenv("COMPANION_DRAFT_REVIEW") or "").strip()
    return _normalize_draft_review_mode(env or _draft_review_mode_from_yaml())


def companion_should_review_draft(major: str, handler: str = "") -> bool:
    mode = companion_draft_review_mode()
    if mode == "off":
        return False
    if mode == "all":
        return True
    if mode == "public":
        return (major or "").strip() == _PUBLIC_KNOWLEDGE_MAJOR
    return False


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

        msgs = [SystemMessage(content=_review_system_prompt()), HumanMessage(content=human)]
        row = invoke_structured_llm(
            llm,
            CompanionReviewOutput,
            msgs,
            label="review_companion_draft",
            fallback=_review_fallback,
        )
        record_llm_call(
            STAGE_REVIEW,
            serialize_langchain_messages(msgs),
            response=row.model_dump_json(by_alias=True),
            label=f"{major} / {intent} · {handler}",
        )
        return _row_to_dict(row)
    except Exception:
        logger.warning("companion review llm failed", exc_info=True)
        return {"pass": True, "reason": "评审调用失败，放行", "hint": ""}
