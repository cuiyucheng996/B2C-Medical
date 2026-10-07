"""B2C 长期记忆：Finding 原始桶 + 抽离后的知识对齐（无 LangGraph 依赖）。"""

from __future__ import annotations

from typing import Literal, TypedDict

FindingKind = Literal["text", "image", "consult_state"]
ExtractKind = Literal["body_part", "disease", "symptom"]
FindingSource = Literal["chat", "upload", "consult"]
CompanionFactKind = Literal[
    "place",
    "time",
    "mood",
    "weather",
    "event",
    "occupation",
    "habit",
    "person",
    "thing",
    "negation",
    "other",
]

__all__ = [
    "CompanionFactKind",
    "CompanionFactRecord",
    "ExtractKind",
    "ExtractedLink",
    "FindingKind",
    "FindingRecord",
    "FindingSource",
    "UserMemoryCard",
]


class ExtractedLink(TypedDict, total=False):
    kind: ExtractKind
    name: str
    mention: str
    node_id: str | None
    label: str
    aligned: bool


class CompanionFactRecord(TypedDict, total=False):
    """陪聊事实（Graphiti 的 fact 形态），不进 Consult* 问诊节点。"""

    id: str
    fact_hash: str
    name: str
    fact: str
    kind: CompanionFactKind
    negated: bool
    last_said: str


class FindingRecord(TypedDict, total=False):
    id: str
    content_hash: str
    kind: FindingKind
    raw_text: str
    image_url: str
    thread_id: str
    source: FindingSource
    created_at: str
    extracts: list[ExtractedLink]


class UserMemoryCard(TypedDict, total=False):
    user_id: str
    summary: str
    current_body_part: str | None
    current_disease: str | None
    current_symptoms: list[str]
    findings: list[FindingRecord]
    companion_facts: list[CompanionFactRecord]
