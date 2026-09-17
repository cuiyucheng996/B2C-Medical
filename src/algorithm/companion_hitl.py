"""C 端 HITL：不靠 LangGraph。钥匙是 user_id + resume_token。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "ask_if_needed",
    "clear_hitl",
    "consume_hitl",
    "get_pending_hitl",
    "save_hitl",
]

_NS = "companion:hitl"
_TTL = 86400  # 关场不清；24 小时无续跑由 Redis 过期删除
_mem: dict[str, dict[str, Any]] = {}
_TASK_HINTS = ("提醒", "打卡", "复查", "随访", "日历", "吃药", "服药", "复诊", "挂号", "报告")

# 具体任务 handler → (缺槽追问话术, agent_name)
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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _key(user_id: str) -> str:
    return f"{_NS}:{(user_id or '').strip()}"


def _client():
    ensure_uvgraph_root()
    from src.configs import REDIS_CONFIG
    from redis import Redis as SyncRedis

    return SyncRedis(
        host=REDIS_CONFIG["host"],
        port=int(REDIS_CONFIG["port"]),
        db=int(REDIS_CONFIG.get("db") or 0),
        password=REDIS_CONFIG.get("password") or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


def save_hitl(
    user_id: str,
    *,
    question: str,
    agent_name: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    uid = (user_id or "").strip()
    token = str(uuid4())
    row = {
        "resume_token": token,
        "pending": True,
        "question": (question or "").strip(),
        "agent_name": (agent_name or "").strip(),
        "state": dict(state or {}),
        "created_at": _now(),
        "user_id": uid,
    }
    if not uid:
        return row
    _mem[uid] = row
    try:
        client = _client()
        try:
            client.set(_key(uid), json.dumps(row, ensure_ascii=False), ex=_TTL)
        finally:
            client.close()
    except Exception:
        logger.warning("save hitl redis failed user_id=%s", uid)
    return row


def get_pending_hitl(user_id: str) -> dict[str, Any] | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    raw = None
    try:
        client = _client()
        try:
            raw = client.get(_key(uid))
        finally:
            client.close()
    except Exception:
        logger.warning("get hitl redis failed user_id=%s", uid)
        raw = None
    if raw:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            row = None
        if isinstance(row, dict) and row.get("pending"):
            _mem[uid] = row
            return dict(row)
    row = _mem.get(uid)
    if isinstance(row, dict) and row.get("pending"):
        return dict(row)
    return None


def consume_hitl(user_id: str, resume_token: str = "") -> dict[str, Any] | None:
    """取出并清除挂起快照。未传 token 则按该用户当前挂起项续跑。"""
    pending = get_pending_hitl(user_id)
    if not pending:
        return None
    token = (resume_token or "").strip()
    expected = str(pending.get("resume_token") or "")
    if token and expected and token != expected:
        return None
    clear_hitl(user_id)
    return pending


def clear_hitl(user_id: str) -> None:
    uid = (user_id or "").strip()
    if not uid:
        return
    _mem.pop(uid, None)
    try:
        client = _client()
        try:
            client.delete(_key(uid))
        finally:
            client.close()
    except Exception:
        logger.warning("clear hitl redis failed user_id=%s", uid)


def ask_if_needed(
    prep: Any,
    *,
    slot_memory: dict[str, list[str]] | None = None,
    has_uploaded_file: bool = False,
) -> tuple[str, str] | None:
    """缺槽则追问。返回 (question, agent_name)，不需要则 None。"""
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
