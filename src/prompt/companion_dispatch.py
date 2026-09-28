"""子图 system：按 major/handler 只注入本路调度说明（不全量贴 companion_tools 调度块）。"""

from __future__ import annotations

from prompt import load_prompt

__all__ = ["build_companion_tools_prompt"]

# 细意图优先（handler 已定时只给本句）
_HANDLER_DISPATCH: dict[str, str] = {
    "mem_session_recall": (
        "不要调任何工具，直接根据上文 messages 里用户原话如实复述；没有历史就如实说还没记录到。"
    ),
    "mem_stated_consult": "只调 lookup_stated_consult，查用户曾提到的部位/疾病/症状。",
    "mem_companion_fact": "只调 lookup_life_facts，查心情、事件、生活事实。",
    "status_complaint": (
        "陈述当前身体不适：结合预处理观察与对话上文组句；可调用 search_long_memory 补长期近况。"
        "不要调公共知识、报告、任务类工具。"
    ),
    "status_life": (
        "陈述生活/心情/天气：可 search_long_memory、lookup_life_facts；不要调问诊图谱与报告工具。"
    ),
    "file_image": "只调 describe_uploaded_image 描述本轮上传的日常图。",
    "file_document": "只调 describe_uploaded_document 描述本轮上传的 Word/Excel/PDF。",
    "report_upload": "只调 explain_uploaded_report（刚上传的 PDF）；不要 explain_mentioned_report / compare_lab_history。",
    "report_tables": "只调 extract_report_tables。",
    "report_explain": "只调 explain_mentioned_report（口头化验单，exam 填检查名）。",
    "report_compare": "只调 compare_lab_history（metric 填指标名，对比历史）。",
    "task_remind": "只调 note_task_remind。任何到点要做的事（复查、赶飞机、办事）都记这一条；口语时间可先 Time MCP 再填 start_iso。不要改去查手册或回答别的医学问题。",
    "task_checkin": "只调 note_task_checkin（打卡）。",
    "task_medication": "只调 note_task_medication（服药）；药名填 drug，时间填 when。",
    "task_appointment": "只调 note_task_appointment（复诊挂号）；医院或科室填 department。",
    "task_report": "只调 note_task_report（取报告/回传报告）；检查项填 exam。",
    "risk_dosage": "只调 safety_policy_hint，勿给具体剂量。",
    "risk_diagnosis": "只调 safety_policy_hint，勿下诊断。",
}

_MAJOR_DISPATCH: dict[str, str] = {
    "公共知识问答": (
        "按本轮已绑定工具调用。部位/病/症/证型只用 query_consult_graph；"
        "search_exam_handbook 只管检查手册；search_chroma_knowledge 只管向量库。"
        "只用 data.items / summary。不要因为绑了一个就三个都调。"
    ),
    "个人记忆问答": (
        "lookup_stated_consult 查部位/病；lookup_life_facts 查心情事件。按细意图选，不要两个都乱调。"
    ),
    "近况陈述": (
        "可 search_long_memory；上传日常图→describe_uploaded_image；"
        "上传 Word/Excel/PDF→describe_uploaded_document。"
    ),
    "报告指标解读": "只调与本路细意图对应的那一个报告工具，不要三个都调，也不要改去调 search_exam_handbook。",
    "具体任务": (
        "只调与本路细意图对应的那一个 note_task_*；不要混用、不要同时调多个。"
        "口语时间可先 Time MCP 得到 ISO。药名 drug，科室 department，检查 exam，时间 when。"
    ),
    "风险拦截": "只调 safety_policy_hint。",
    "闲聊": "通常无工具；没有绑定就不要强行调工具。",
}

_KG_HANDLERS = frozenset(
    {
        "kg_part_diseases",
        "kg_disease_symptoms",
        "kg_symptom_parts",
        "kg_disease_plain",
        "kg_syndrome_signs",
        "kg_exam_meaning",
        "kg_lab_range",
        "kg_symptom_care",
    }
)

_MCP_HINT = (
    "若本轮还绑定了 MCP：Time 补日期；Fetch 抓公开页；百炼 bailian_retrieve 补运营/合规；"
    "web_search 补政策科普；oss_presign_upload 仅报告直传 OSS；graphiti 补长期事实。"
    "不要 add_memory、不要清图、不要把网页/联网结果当诊断。"
)

_LOCAL_TOOL_NAMES = frozenset(
    {
        "lookup_stated_consult",
        "lookup_life_facts",
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


def _dispatch_lines(major: str, handler: str) -> str:
    h = (handler or "").strip()
    m = (major or "").strip()
    if h and h in _HANDLER_DISPATCH:
        return _HANDLER_DISPATCH[h]
    if h in _KG_HANDLERS:
        if h in {"kg_exam_meaning", "kg_lab_range"}:
            return (
                "检查/化验含义或参考范围：按绑定顺序用 search_exam_handbook 或 search_chroma_knowledge，"
                "不要同时乱调 query_consult_graph。"
            )
        if h == "kg_symptom_care":
            return "生活注意事项：用 search_exam_handbook；只用 data.items / summary。"
        return "公共知识：只用 query_consult_graph；query 填标准名/症状/证型相关检索词。"
    if m and m in _MAJOR_DISPATCH:
        return _MAJOR_DISPATCH[m]
    return "只调本轮已绑定工具；证据够了就停止调用。"


def build_companion_tools_prompt(
    *,
    major: str = "",
    intent: str = "",
    handler: str = "",
    bound_tools: list[str] | None = None,
) -> str:
    head = load_prompt("companion_tools_head")
    route = " / ".join(p for p in (major, intent, f"（{handler}）" if handler else "") if p)
    dispatch = _dispatch_lines(major, handler)
    names = [str(n or "").strip() for n in (bound_tools or []) if str(n or "").strip()]
    names = [n for n in names if n != "CompanionReply"]
    tool_line = ""
    if names:
        tool_line = "本轮已绑定工具（只能从中选）：" + "、".join(names)
    parts = [head, f"本路调度（{route}）：\n{dispatch}"]
    if tool_line:
        parts.append(tool_line)
    if names and any(n not in _LOCAL_TOOL_NAMES for n in names):
        parts.append(_MCP_HINT)
    parts.append("没有绑定工具时不要强行调工具。证据够了就停止调用工具。")
    return "\n\n".join(parts)
