"""C 端意图清单：大类 + 细意图名（细名与老师一样，是下游模板主键）。"""

from __future__ import annotations

from typing import Any, Literal

MajorIntent = Literal[
    "公共知识问答",
    "具体任务",
    "个人记忆问答",
    "报告指标解读",
    "风险拦截",
    "近况陈述",
    "闲聊",
    "other",
]

# 细意图：名字必须与训练 CSV、规则表、match 分支一致
INTENT_ROWS: tuple[dict[str, Any], ...] = (
    # —— 公共知识问答（对照老师「查询某商品…」细拆图模式）——
    {
        "intent": "查询某部位有哪些常见疾病",
        "major": "公共知识问答",
        "slots": ("部位",),
        "handler": "kg_part_diseases",
    },
    {
        "intent": "查询某疾病常见有哪些症状",
        "major": "公共知识问答",
        "slots": ("疾病",),
        "handler": "kg_disease_symptoms",
    },
    {
        "intent": "查询某症状常见出现在哪些部位",
        "major": "公共知识问答",
        "slots": ("症状",),
        "handler": "kg_symptom_parts",
    },
    {
        "intent": "查询某疾病的通俗解释",
        "major": "公共知识问答",
        "slots": ("疾病",),
        "handler": "kg_disease_plain",
    },
    {
        "intent": "查询某检查项目是检查什么的",
        "major": "公共知识问答",
        "slots": ("检查",),
        "handler": "kg_exam_meaning",
    },
    {
        "intent": "查询某指标的正常范围或含义",
        "major": "公共知识问答",
        "slots": ("指标",),
        "handler": "kg_lab_range",
    },
    {
        "intent": "查询某症状的生活注意事项",
        "major": "公共知识问答",
        "slots": ("症状",),
        "handler": "kg_symptom_care",
    },
    {
        "intent": "查询某证型常见有哪些表现",
        "major": "公共知识问答",
        "slots": ("证型",),
        "handler": "kg_syndrome_signs",
    },
    # —— 其余大类：少量细分，先能跑通路由 ——
    {
        "intent": "提醒用户某日复查或随访",
        "major": "具体任务",
        "slots": ("时间",),
        "handler": "task_remind",
    },
    {
        "intent": "记录用户完成一次打卡",
        "major": "具体任务",
        "slots": (),
        "handler": "task_checkin",
    },
    {
        "intent": "提醒用户按时服药",
        "major": "具体任务",
        "slots": ("时间",),
        "handler": "task_medication",
    },
    {
        "intent": "预约复诊或挂号",
        "major": "具体任务",
        "slots": ("时间",),
        "handler": "task_appointment",
    },
    {
        "intent": "提醒回传或领取检查报告",
        "major": "具体任务",
        "slots": ("检查", "时间"),
        "handler": "task_report",
    },
    {
        "intent": "查询用户曾经提到的部位或疾病",
        "major": "个人记忆问答",
        "slots": (),
        "handler": "mem_stated_consult",
    },
    {
        "intent": "查询用户最近的心情或事件",
        "major": "个人记忆问答",
        "slots": (),
        "handler": "mem_companion_fact",
    },
    {
        "intent": "解读用户提到的检查报告",
        "major": "报告指标解读",
        "slots": ("检查",),
        "handler": "report_explain",
    },
    {
        "intent": "上传检查报告并解读",
        "major": "报告指标解读",
        "slots": ("检查",),
        "handler": "report_upload",
    },
    {
        "intent": "对比某指标与历史记录",
        "major": "报告指标解读",
        "slots": ("指标",),
        "handler": "report_compare",
    },
    {
        "intent": "解读用户上传的日常图片",
        "major": "近况陈述",
        "slots": (),
        "handler": "file_image",
    },
    {
        "intent": "解读用户上传的普通文档",
        "major": "近况陈述",
        "slots": (),
        "handler": "file_document",
    },
    {
        "intent": "拦截急症或求救描述",
        "major": "风险拦截",
        "slots": (),
        "handler": "risk_emergency",
    },
    {
        "intent": "拦截具体用药剂量咨询",
        "major": "风险拦截",
        "slots": (),
        "handler": "risk_dosage",
    },
    {
        "intent": "陈述当前身体不适",
        "major": "近况陈述",
        "slots": ("症状",),
        "handler": "status_complaint",
    },
    {
        "intent": "陈述生活事件心情或天气",
        "major": "近况陈述",
        "slots": (),
        "handler": "status_life",
    },
    {
        "intent": "寒暄打招呼",
        "major": "闲聊",
        "slots": (),
        "handler": "chat_hello",
    },
    {
        "intent": "感谢或结束对话",
        "major": "闲聊",
        "slots": (),
        "handler": "chat_thanks",
    },
)

INTENT_BY_NAME: dict[str, dict[str, Any]] = {row["intent"]: row for row in INTENT_ROWS}

# 给第三方模型看的短说明；易混的才写，避免提示过长
INTENT_HINTS: dict[str, str] = {
    "查询某症状的生活注意事项": (
        "症状的护理、现场处理、止血、忌口、能不能运动、饮食注意。"
        "例：「发生创伤出血时怎么止血」「头痛要注意什么」。不要标未细分。"
    ),
    "拦截急症或求救描述": (
        "仅限明确救命、叫救护车、晕倒、剧烈胸痛。"
        "问怎么止血/怎么护理属于「查询某症状的生活注意事项」，不是本条。"
    ),
    "查询某疾病常见有哪些症状": "问某病有哪些表现，不是问出血怎么处理。",
}

__all__ = ["INTENT_BY_NAME", "INTENT_HINTS", "INTENT_ROWS", "MajorIntent", "major_of", "intent_prompt_lines"]


def intent_prompt_lines(*, major: str | None = None) -> str:
    lines: list[str] = []
    for row in INTENT_ROWS:
        name = str(row["intent"])
        if major and str(row.get("major") or "") != major:
            continue
        hint = INTENT_HINTS.get(name, "")
        extra = f"：{hint}" if hint else ""
        if major:
            lines.append(f"- {name}{extra}")
        else:
            lines.append(f"- {name}（大类：{row['major']}）{extra}")
    return "\n".join(lines)


def major_of(intent: str) -> str:
    row = INTENT_BY_NAME.get(intent or "")
    if row:
        return str(row["major"])
    if intent in ("other", "未知意图", ""):
        return "other"
    return "other"
