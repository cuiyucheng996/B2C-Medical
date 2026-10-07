"""Consultation data schemas (literals and structured payloads).

Pure type definitions with no LangGraph dependencies.
"""

from __future__ import annotations

from typing import Literal, TypedDict

MajorRound = Literal[1, 2, 3, 4]
"""第一大轮=1 部位+疾病；第二大轮=2 症状；第三大轮=3 补充；第四大轮=4 快捷补充。"""

UserInputMode = Literal["option", "text", "number"]
"""用户输入方式：点选项、自由说话、回复数字。"""

InputChannel = Literal["text", "asr"]
"""入口通道：键盘打字 vs 语音识别（ASR）。"""

DEFAULT_INPUT_CHANNEL: InputChannel = "text"

BodyPart = Literal[
    "头面部",
    "颈部",
    "胸胁部",
    "胃腹部",
    "二阴部",
    "四肢躯干",
    "其他",
    "头面部、四肢躯干",
]
"""八大部位，需与问诊 UI / 疾病映射保持一致。"""

OTHER_OPTION = "找不到对应部位"
"""UI 兜底选项：表示用户需改用文字描述，不是有效部位/疾病值。"""

SYMPTOM_OTHER_OPTION = "没有太像的"
"""第二大轮 UI 兜底：表示选项都不符合，需文字描述具体症状。"""

UiBodySelection = BodyPart | Literal["找不到对应部位"]
"""部位 UI 选项（含八大部位与兜底项）。"""

UiSymptomSelection = str
"""症状 UI 选项：标准症状名或 SYMPTOM_OTHER_OPTION。"""


def is_other_option(value: str | None) -> bool:
    return value == OTHER_OPTION


def is_symptom_other_option(value: str | None) -> bool:
    return value == SYMPTOM_OTHER_OPTION


def is_valid_body_selection(value: str | None) -> bool:
    return bool(value) and not is_other_option(value)


def is_valid_disease_selection(value: str | None) -> bool:
    return bool(value) and not is_other_option(value)


def is_valid_symptom_selection(value: str | None) -> bool:
    return bool(value) and not is_symptom_other_option(value)


def parse_input_channel(
    value: str | None,
    *,
    default: InputChannel = DEFAULT_INPUT_CHANNEL,
) -> InputChannel:
    """解析 HTTP body / Header 中的入口通道标识。"""
    normalized = (value or "").strip().lower()
    if normalized in ("text", "typing", "keyboard", "type"):
        return "text"
    if normalized in ("asr", "voice", "speech", "stt"):
        return "asr"
    return default

NextAction = Literal[
    "await_user",
    "remind_relevance",
    "retry_round1",
    "emit_protocol",
    "present_top3",
    "finalize",
]
"""路由提示，供条件边读取。"""

SessionMode = Literal[
    "idle",
    "await_body",
    "await_disease",
    "await_symptom",
    "await_consent",
    "inference_run",
]
"""对外业务模式：当前会话允许哪些用户输入。"""

UserConsent = Literal["consent", "decline", "unclear"]
"""用户对继续分析的回应。"""


class InferencePayload(TypedDict, total=False):
    """最终交给推理机的结构化结果。"""

    body_part: str
    disease_type: str
    confirmed_symptom: str
    syndrome: str
    supplementary_symptoms: list[str]
    protocol_round1_emitted: bool
