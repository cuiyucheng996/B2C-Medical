"""陪聊回复动作条：槽齐之后的快捷操作，不是补槽。"""

from __future__ import annotations

from typing import Any

__all__ = ["reply_actions"]

_UPLOAD = {"id": "upload_report", "label": "上传化验单"}
_BOOK = {"id": "book_expert", "label": "预约专家问诊"}


def reply_actions(
    *,
    hitl: dict[str, Any] | None = None,
    preprocess: dict[str, Any] | None = None,
    last_report: dict[str, Any] | None = None,
    slot_memory: dict[str, list[str]] | None = None,
) -> list[dict[str, str]]:
    """HITL 补槽不出按钮；等上传文件、或槽已齐的知识/报告解读才出。"""
    prep = preprocess if isinstance(preprocess, dict) else {}
    major = str(prep.get("intent_major") or "")
    handler = str(prep.get("intent_handler") or "")
    mentions = prep.get("mentions") if isinstance(prep.get("mentions"), dict) else {}
    slots = slot_memory if isinstance(slot_memory, dict) else {}
    waiting_upload = bool(hitl) and str(hitl.get("agent_name") or "") == "explain_uploaded_report"
    if hitl and not waiting_upload:
        return []

    actions: list[dict[str, str]] = []
    reportish = major == "报告指标解读" or handler.startswith("report_")
    if waiting_upload or (not hitl and reportish):
        actions.append(dict(_UPLOAD))

    if waiting_upload:
        return actions

    has_topic = bool(
        mentions.get("部位")
        or mentions.get("疾病")
        or mentions.get("症状")
        or slots.get("部位")
        or slots.get("疾病")
        or slots.get("症状")
        or last_report
    )
    if has_topic and major in ("公共知识问答", "报告指标解读", "近况陈述"):
        actions.append(dict(_BOOK))
    return actions
