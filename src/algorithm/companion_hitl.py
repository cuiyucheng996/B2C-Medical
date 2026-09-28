"""C 端 HITL：待办状态在 LangGraph checkpointer（见 hitl_checkpoint）。"""

from __future__ import annotations

from typing import Any

from algorithm.hitl_checkpoint import (
    HITL_KIND_SLOT,
    HITL_KIND_TOOL,
    clear_session_hitl,
    consume_slot_hitl,
    get_pending_hitl,
    pop_tool_pending,
    put_slot_pending,
    put_tool_pending,
    read_tool_pending,
    slot_thread_id,
)

__all__ = [
    "HITL_KIND_SLOT",
    "HITL_KIND_TOOL",
    "ask_if_needed",
    "build_tool_hitl_resume",
    "clear_hitl",
    "consume_hitl",
    "format_tool_hitl_question",
    "get_pending_hitl",
    "parse_client_hitl_decision",
    "save_hitl",
    "save_tool_hitl",
    "slot_thread_id",
]

_TASK_HINTS = ("提醒", "打卡", "复查", "随访", "日历", "吃药", "服药", "复诊", "挂号", "报告")

_TASK_ASK: dict[str, tuple[str, str]] = {
    "task_remind": ("复查或随访大概安排在哪一天？", "note_task_remind"),
    "task_checkin": ("这次打卡完成的是哪件事？", "note_task_checkin"),
    "task_medication": ("这个药医生让你什么时候吃？比如早饭后、睡前。", "note_task_medication"),
    "task_appointment": ("复诊是哪个医院或科室，大概哪一天？", "note_task_appointment"),
    "task_report": ("是哪项检查的报告，大概什么时候能取？", "note_task_report"),
}
_TASK_ASK_DEFAULT = ("你是想设复查提醒、服药提醒、复诊预约，还是取报告提醒？大概什么时间？", "note_task_remind")
_REPORT_HINTS = ("报告", "化验", "体检", "检查单", "验血", "【上传检查报告】")
_METRIC_HINTS = ("糖", "胆固醇", "转氨酶", "血压", "尿酸", "甘油三酯", "血红蛋白", "alt", "ast", "ldl", "hdl")


def save_hitl(
    session_id: str,
    *,
    question: str,
    agent_name: str,
    state: dict[str, Any],
    hitl_kind: str = HITL_KIND_SLOT,
    action_requests: list[dict[str, Any]] | None = None,
    review_configs: list[dict[str, Any]] | None = None,
    thread_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """挂起待办。session_id 必填；user_id 仅兼容旧调用，可忽略。"""
    sid = (session_id or "").strip()
    if str(hitl_kind or "") == HITL_KIND_TOOL:
        row = {
            "resume_token": (thread_id or "").strip(),
            "thread_id": (thread_id or "").strip(),
            "pending": True,
            "hitl_kind": HITL_KIND_TOOL,
            "question": (question or "").strip(),
            "agent_name": (agent_name or "").strip(),
            "state": dict(state or {}),
            "action_requests": list(action_requests or []),
            "review_configs": list(review_configs or []),
            "session_id": sid,
            "user_id": (user_id or "").strip(),
        }
        put_tool_pending(sid, row)
        return row
    return put_slot_pending(
        sid,
        question=question,
        agent_name=agent_name,
        state=state,
    )


def save_tool_hitl(session_id: str, row: dict[str, Any]) -> dict[str, Any]:
    put_tool_pending(session_id, row)
    return row


def consume_hitl(
    session_id: str,
    resume_token: str = "",
    *,
    kinds: tuple[str, ...] | None = None,
    user_text: str = "",
) -> dict[str, Any] | None:
    sid = (session_id or "").strip()
    if kinds and HITL_KIND_TOOL in kinds and HITL_KIND_SLOT not in kinds:
        pending = read_tool_pending(sid)
        if not pending:
            return None
        token = (resume_token or "").strip()
        expected = str(pending.get("resume_token") or pending.get("thread_id") or "")
        if token and expected and token != expected:
            return None
        return pop_tool_pending(sid)
    if kinds and HITL_KIND_SLOT not in (kinds or (HITL_KIND_SLOT,)):
        return None
    return consume_slot_hitl(sid, resume_token, user_text)


def clear_hitl(session_id: str) -> None:
    clear_session_hitl(session_id)


def format_tool_hitl_question(hitl_request: dict[str, Any]) -> str:
    import json

    actions = list(hitl_request.get("action_requests") or [])
    if not actions:
        return "请确认是否执行待审批的工具调用。"
    lines = ["请确认是否执行以下操作（可批准 / 修改参数 / 拒绝 / 用文字说明）："]
    for item in actions:
        name = str(item.get("name") or "")
        args = item.get("args") or {}
        lines.append(f"· {name}：{json.dumps(args, ensure_ascii=False)}")
        desc = str(item.get("description") or "").strip()
        if desc:
            lines.append(f"  {desc}")
    return "\n".join(lines)


def build_tool_hitl_resume(
    pending: dict[str, Any],
    *,
    decision: dict[str, Any] | None,
    user_text: str = "",
) -> dict[str, Any]:
    n = len(pending.get("action_requests") or [])
    if n < 1:
        n = 1
    decisions = parse_client_hitl_decision(decision, user_text=user_text, count=n)
    return {"decisions": decisions}


def parse_client_hitl_decision(
    decision: dict[str, Any] | None,
    *,
    user_text: str = "",
    count: int = 1,
) -> list[dict[str, Any]]:
    text = (user_text or "").strip()
    body = decision if isinstance(decision, dict) else {}
    dtype = str(body.get("type") or body.get("decision") or "").strip().lower()
    if not dtype and text in {"确认", "批准", "同意", "ok", "OK"}:
        dtype = "approve"
    if not dtype and text:
        dtype = "respond"
    if not dtype:
        dtype = "approve"
    out: list[dict[str, Any]] = []
    for _ in range(max(1, count)):
        if dtype == "approve":
            out.append({"type": "approve"})
        elif dtype == "edit":
            edited = body.get("edited_action")
            if not isinstance(edited, dict):
                edited = body.get("edited_args") if isinstance(body.get("edited_args"), dict) else {}
            name = str(body.get("tool_name") or body.get("name") or edited.get("name") or "")
            args = edited.get("args") if isinstance(edited.get("args"), dict) else edited
            if not isinstance(args, dict):
                args = {}
            out.append({"type": "edit", "edited_action": {"name": name, "args": args}})
        elif dtype == "reject":
            msg = str(body.get("message") or text or "用户已拒绝执行该工具。").strip()
            out.append({"type": "reject", "message": msg})
        else:
            msg = str(body.get("message") or text or "用户补充说明").strip()
            out.append({"type": "respond", "message": msg})
    return out


def ask_if_needed(
    prep: Any,
    *,
    slot_memory: dict[str, list[str]] | None = None,
    has_uploaded_file: bool = False,
) -> tuple[str, str] | None:
    major = str(getattr(prep, "intent_major", "") or "")
    text = str(getattr(prep, "resolved_text", "") or getattr(prep, "corrected_text", "") or "")
    mentions = getattr(prep, "mentions", None)
    mentions = mentions if isinstance(mentions, dict) else {}
    slots = slot_memory if isinstance(slot_memory, dict) else {}
    alignments = list(getattr(prep, "alignments", None) or [])
    if major == "具体任务":
        handler = str(getattr(prep, "intent_handler", "") or "")
        if not any(hint in text for hint in _TASK_HINTS):
            question, agent_name = _TASK_ASK.get(handler, _TASK_ASK_DEFAULT)
            return question, agent_name
        return None
    if major == "公共知识问答":
        has_topic = bool(
            mentions.get("部位")
            or mentions.get("疾病")
            or mentions.get("症状")
            or slots.get("部位")
            or slots.get("疾病")
            or slots.get("症状")
            or alignments
        )
        if not has_topic:
            return "请问你想了解哪个部位、哪种病，还是哪项检查？", "query_consult_graph"
        return None
    if major == "报告指标解读":
        handler = str(getattr(prep, "intent_handler", "") or "")
        lowered = text.lower()
        if handler == "report_upload" and not has_uploaded_file and "上传检查报告" not in text.replace("【", "").replace("】", ""):
            return "请先上传检查报告 PDF 或化验单照片。", "explain_uploaded_report"
        if handler == "report_explain" and not any(hint in text for hint in _REPORT_HINTS):
            return "你想看哪一份检查报告？比如血常规、肝功能。", "explain_mentioned_report"
        if handler == "report_compare":
            has_metric = bool(mentions.get("指标")) or any(hint in lowered for hint in _METRIC_HINTS)
            if not has_metric:
                return "要对比哪一项指标？比如胆固醇、空腹血糖。", "compare_lab_history"
        return None
    return None
