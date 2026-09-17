"""按用户缓存 ConversationEntityMemory / ConversationKGMemory，供中间件调用。"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

__all__ = ["clear_entity_kg_memory", "recall_and_update"]

_LOCK = Lock()
_CACHE: dict[str, tuple[Any, Any]] = {}
_SKIP_MARKS = ("【预处理观察】", "【实体记忆】", "【思考】", "【工具甄别】")


def clear_entity_kg_memory(user_key: str = "") -> None:
    key = (user_key or "").strip()
    with _LOCK:
        if not key:
            _CACHE.clear()
            return
        _CACHE.pop(key, None)


def _history_for_memory(messages: list[Any]) -> list[BaseMessage]:
    rows: list[BaseMessage] = []
    for msg in messages or []:
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        text = str(getattr(msg, "content", "") or "").strip()
        if not text or any(mark in text for mark in _SKIP_MARKS):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        rows.append(HumanMessage(content=text) if isinstance(msg, HumanMessage) else AIMessage(content=text))
    return rows[-12:]


def _llm() -> Any:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.llm.chat import get_chat_model

    return get_chat_model()


def _pair(user_key: str, llm: Any) -> tuple[Any, Any]:
    with _LOCK:
        cached = _CACHE.get(user_key)
        if cached is not None:
            return cached
        from langchain_classic.memory.entity import ConversationEntityMemory
        from langchain_community.memory.kg import ConversationKGMemory

        entity = ConversationEntityMemory(
            llm=llm,
            k=3,
            input_key="input",
            output_key="output",
            return_messages=False,
        )
        kg = ConversationKGMemory(
            llm=llm,
            k=2,
            input_key="input",
            output_key="output",
            return_messages=False,
        )
        _CACHE[user_key] = (entity, kg)
        return entity, kg


def _fill_buffer(memory: Any, history: list[BaseMessage]) -> None:
    chat = getattr(memory, "chat_memory", None)
    if chat is None:
        return
    chat.clear()
    for msg in history:
        chat.add_message(msg)


def _format_note(entity_vars: dict[str, Any], kg_vars: dict[str, Any], cache: list[str]) -> str:
    lines = ["【实体记忆】"]
    names = [str(name).strip() for name in cache if str(name).strip() and str(name).strip().upper() != "NONE"]
    if names:
        lines.append("本轮实体：" + "、".join(names))
    summaries = entity_vars.get("entities") if isinstance(entity_vars, dict) else None
    if isinstance(summaries, dict):
        bits = []
        for name, summary in summaries.items():
            text = str(summary or "").strip()
            if text:
                bits.append(f"{name}：{text}")
        if bits:
            lines.append("实体摘要：" + "；".join(bits[:12]))
    kg_text = ""
    if isinstance(kg_vars, dict):
        raw = kg_vars.get("history")
        if isinstance(raw, str):
            kg_text = raw.strip()
        elif isinstance(raw, list):
            kg_text = "\n".join(str(getattr(item, "content", item) or "") for item in raw).strip()
    if kg_text:
        lines.append("图谱关系：\n" + kg_text)
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def recall_and_update(
    query: str,
    messages: list[Any],
    *,
    user_key: str,
) -> str:
    """抽取实体摘要 + 三元组写入内存店，返回给模型看的附注。失败返回空串。"""
    text = (query or "").strip()
    key = (user_key or "").strip() or "_anon"
    if not text:
        return ""
    try:
        llm = _llm()
        if llm is None:
            return ""
        entity, kg = _pair(key, llm)
        history = _history_for_memory(messages)
        _fill_buffer(entity, history)
        _fill_buffer(kg, history)
        payload = {"input": text}
        entity_vars = entity.load_memory_variables(payload)
        kg_vars = kg.load_memory_variables(payload)
        dummy = {"output": "ok"}
        entity.save_context(payload, dummy)
        kg.save_context(payload, dummy)
        return _format_note(entity_vars, kg_vars, list(getattr(entity, "entity_cache", None) or []))
    except Exception:
        logger.warning("ConversationEntityMemory/KG recall failed user=%s", key, exc_info=True)
        return ""
