"""医疗问诊 FAQ 类型与业务字面量（无 LangGraph 依赖）。"""

from __future__ import annotations

from typing import Literal, TypedDict

FaqCategory = Literal[
    "usage",       # 产品使用
    "process",     # 流程说明
    "disclaimer",  # 合规免责
    "chitchat",    # 寒暄引导
    "knowledge",   # 轻量中医/范围知识
    "boundary",    # 拒答边界
]

FaqPhase = Literal[
    "any",     # 任意阶段
    "body",    # 等待选部位
    "disease", # 等待选疾病
    "consent", # 等待确认继续
    "idle",    # 无 interrupt / 流程结束
]

FaqAction = Literal[
    "reply_only",          # 答完继续等待
    "reply_and_continue",  # 答完仍停当前节点
    "disclaimer",          # 带免责，不给诊断/用药
]

FAQ_CATEGORIES: tuple[FaqCategory, ...] = (
    "usage",
    "process",
    "disclaimer",
    "chitchat",
    "knowledge",
    "boundary",
)

FAQ_PHASES: tuple[FaqPhase, ...] = (
    "any",
    "body",
    "disease",
    "consent",
    "idle",
)


class MedicalFaqItem(TypedDict):
    id: str
    question: str
    answer: str
    category: FaqCategory
    keywords: str
    phase: FaqPhase
    priority: int
    action: FaqAction


__all__ = [
    "FAQ_CATEGORIES",
    "FAQ_PHASES",
    "FaqAction",
    "FaqCategory",
    "FaqPhase",
    "MedicalFaqItem",
]
