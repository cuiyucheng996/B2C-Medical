"""医疗 Agent 工厂：LLM 链与 system 提示词统一维护入口。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from ...llm.chat import get_chat_model

LANGCHAIN_ROOT = Path(__file__).resolve().parents[1]

__all__ = [
    "get_medical_agent",
    "get_medical_chain",
    "load_medical_system_prompt",
]


def _load_prompt(name: str) -> str:
    return (LANGCHAIN_ROOT / "prompt" / name).read_text(encoding="utf-8").strip()


def load_medical_system_prompt() -> str:
    """医疗问诊 system 提示词（单一维护入口）。"""
    return _load_prompt("medical_dialog_prompt.txt")


@lru_cache
def get_medical_chain() -> BaseChatModel | None:
    """医疗对话模型：直接接收标准 ``messages`` 列表（system / human / ai）。"""
    return get_chat_model()


def get_medical_agent() -> BaseChatModel | None:
    """兼容旧名。"""
    return get_medical_chain()
