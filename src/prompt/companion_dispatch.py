"""子图 system：按 major/handler 注入 intent_skills 中的 SOP（不全量贴 companion_tools 调度块）。"""

from __future__ import annotations

from functools import lru_cache

from prompt import load_prompt
from prompt.intent_skill_loader import load_intent_skill, load_major_skill

__all__ = [
    "build_companion_route_prompt",
    "build_companion_route_tail",
    "build_companion_skill_prompt",
    "build_companion_tools_prompt",
]

_MCP_HINT = (
    "若本轮还绑定了 MCP：Time 补日期；Fetch 抓公开页；百炼 bailian_retrieve 补运营/合规；"
    "web_search（通义）与 brave_web_search（Brave）在本地 handbook/chroma 仍不够时补公开科普；"
    "oss_presign_upload 仅报告直传 OSS；graphiti 补长期事实；"
    "影刀 queryApplist/queryRobotParam 查流程参数，startJob/runApp 启动作业后用 queryJob 查状态，勿编造。"
    "不要 add_memory、不要清图、不要把网页/联网结果当诊断。"
)

_LOCAL_TOOL_NAMES = frozenset(
    {
        "lookup_stated_consult",
        "lookup_life_facts",
        "query_personal_memory",
        "query_consult_graph",
        "search_exam_handbook",
        "search_chroma_knowledge",
        "search_long_memory",
        "note_task_remind",
        "note_task_checkin",
        "note_task_medication",
        "note_task_appointment",
        "note_task_report",
        "explain_uploaded_report",
        "extract_report_tables",
        "explain_mentioned_report",
        "compare_lab_history",
        "describe_uploaded_image",
        "describe_uploaded_document",
        "safety_policy_hint",
        "CompanionReply",
    }
)

_DEFAULT_DISPATCH = "只调本轮已绑定工具；证据够了就停止调用。"


def _dispatch_lines(major: str, handler: str) -> str:
    h = (handler or "").strip()
    m = (major or "").strip()
    if h:
        skill = load_intent_skill(h)
        if skill:
            return skill
    if m:
        major_skill = load_major_skill(m)
        if major_skill:
            return major_skill
    return _DEFAULT_DISPATCH


@lru_cache(maxsize=256)
def _build_companion_skill_cached(major: str, handler: str) -> str:
    m = (major or "").strip()
    h = (handler or "").strip()
    dispatch = _dispatch_lines(m, h)
    if h:
        label = f"Skill / 调度（{m} / （{h}））" if m else f"Skill / 调度（（{h}））"
    elif m:
        label = f"Skill / 调度（{m}）"
    else:
        label = "Skill / 调度"
    return f"{label}：\n{dispatch}"


@lru_cache(maxsize=256)
def _build_companion_route_tail_cached(
    major: str,
    intent: str,
    handler: str,
    bound_tools_key: str,
) -> str:
    route = " / ".join(p for p in (major, intent, f"（{handler}）" if handler else "") if p)
    names = [n for n in bound_tools_key.split("\x1f") if n and n != "CompanionReply"]
    tool_line = ""
    if names:
        tool_line = "本轮已绑定工具（只能从中选）：" + "、".join(names)
    parts: list[str] = []
    if route:
        parts.append(f"本路标识：{route}")
    if tool_line:
        parts.append(tool_line)
    if names and any(n not in _LOCAL_TOOL_NAMES for n in names):
        parts.append(_MCP_HINT)
    parts.append("没有绑定工具时不要强行调工具。证据够了就停止调用工具。")
    return "\n\n".join(parts)


def build_companion_skill_prompt(*, major: str = "", handler: str = "") -> str:
    return _build_companion_skill_cached((major or "").strip(), (handler or "").strip())


def build_companion_route_tail(
    *,
    major: str = "",
    intent: str = "",
    handler: str = "",
    bound_tools: list[str] | None = None,
) -> str:
    names = [str(n or "").strip() for n in (bound_tools or []) if str(n or "").strip()]
    names = [n for n in names if n != "CompanionReply"]
    key = "\x1f".join(sorted(names))
    return _build_companion_route_tail_cached(
        (major or "").strip(),
        (intent or "").strip(),
        (handler or "").strip(),
        key,
    )


@lru_cache(maxsize=256)
def _build_companion_route_prompt_cached(
    major: str,
    intent: str,
    handler: str,
    bound_tools_key: str,
) -> str:
    skill = _build_companion_skill_cached(major, handler)
    tail = _build_companion_route_tail_cached(major, intent, handler, bound_tools_key)
    return "\n\n".join(p for p in (skill, tail) if p)


def build_companion_route_prompt(
    *,
    major: str = "",
    intent: str = "",
    handler: str = "",
    bound_tools: list[str] | None = None,
) -> str:
    names = [str(n or "").strip() for n in (bound_tools or []) if str(n or "").strip()]
    names = [n for n in names if n != "CompanionReply"]
    key = "\x1f".join(sorted(names))
    return _build_companion_route_prompt_cached(
        (major or "").strip(),
        (intent or "").strip(),
        (handler or "").strip(),
        key,
    )


@lru_cache(maxsize=1)
def _companion_tools_head_cached() -> str:
    return load_prompt("companion_tools_head")


@lru_cache(maxsize=256)
def _build_companion_tools_prompt_cached(
    major: str,
    intent: str,
    handler: str,
    bound_tools_key: str,
) -> str:
    head = _companion_tools_head_cached()
    route_body = _build_companion_route_prompt_cached(major, intent, handler, bound_tools_key)
    return f"{head}\n\n{route_body}"


def build_companion_tools_prompt(
    *,
    major: str = "",
    intent: str = "",
    handler: str = "",
    bound_tools: list[str] | None = None,
) -> str:
    names = [str(n or "").strip() for n in (bound_tools or []) if str(n or "").strip()]
    names = [n for n in names if n != "CompanionReply"]
    key = "\x1f".join(sorted(names))
    return _build_companion_tools_prompt_cached(
        (major or "").strip(),
        (intent or "").strip(),
        (handler or "").strip(),
        key,
    )
