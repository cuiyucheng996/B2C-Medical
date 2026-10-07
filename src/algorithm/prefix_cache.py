"""Companion LLM 前缀缓存：稳定 system 块打 cache_control，并按厂商附加 extra_body。"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import SystemMessage

__all__ = [
    "cache_mark_message",
    "cached_system_message",
    "companion_layered_system_message",
    "companion_prefix_cache_enabled",
    "companion_split_system_message",
    "wrap_companion_chat_model",
]


def companion_prefix_cache_enabled() -> bool:
    raw = (os.environ.get("COMPANION_PREFIX_CACHE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content or "")


def _content_has_cache_control(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("cache_control") for block in content)


def cache_mark_message(msg: Any) -> Any:
    """单条 message 正文标 ephemeral，供子 Agent 多轮 model 复用前缀。"""
    if msg is None or not companion_prefix_cache_enabled():
        return msg
    content = getattr(msg, "content", "")
    if _content_has_cache_control(content):
        return msg
    text = _content_to_str(content).strip()
    if not text:
        return msg
    marked = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    copy_fn = getattr(msg, "model_copy", None)
    if callable(copy_fn):
        try:
            return copy_fn(update={"content": marked})
        except TypeError:
            pass
    try:
        return msg.__class__(content=marked)
    except Exception:
        return msg


def _cached_text_block(text: str) -> dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }


def companion_layered_system_message(
    static_text: str,
    skill_text: str,
    dynamic_text: str,
) -> SystemMessage:
    """全局静态 + 按 handler 的 Skill 打 cache_control；本路 intent/工具等后缀不缓存。"""
    static = (static_text or "").strip()
    skill = (skill_text or "").strip()
    dynamic = (dynamic_text or "").strip()
    if not static and not skill and not dynamic:
        return SystemMessage(content="")
    if not companion_prefix_cache_enabled():
        body = "\n\n".join(part for part in (static, skill, dynamic) if part)
        return SystemMessage(content=body)
    blocks: list[dict[str, Any]] = []
    if static:
        blocks.append(_cached_text_block(static))
    if skill:
        blocks.append(_cached_text_block(skill))
    if dynamic:
        blocks.append({"type": "text", "text": dynamic})
    return SystemMessage(content=blocks)


def companion_split_system_message(static_text: str, dynamic_text: str) -> SystemMessage:
    """稳定前缀打 cache_control，每路子任务后缀不缓存，便于并行多路共享前缀。"""
    return companion_layered_system_message(static_text, "", dynamic_text)


def cached_system_message(static_text: str) -> SystemMessage:
    """将稳定 system 正文标为 ephemeral，供 OpenAI 兼容前缀缓存命中。"""
    body = (static_text or "").strip()
    if not body or not companion_prefix_cache_enabled():
        return SystemMessage(content=body)
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": body,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    )


def _provider_extra_body() -> dict[str, Any] | None:
    """可选 JSON：COMPANION_LLM_EXTRA_BODY='{"prompt_cache_key":"companion-v1"}'。"""
    raw = (os.environ.get("COMPANION_LLM_EXTRA_BODY") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    if not companion_prefix_cache_enabled():
        return None
    # Moonshot / 部分 OpenAI 兼容：稳定 key 便于服务端复用前缀
    return {"prompt_cache_key": "companion-b2c-v1"}


def wrap_companion_chat_model(model: Any) -> Any:
    if model is None:
        return None
    extra = _provider_extra_body()
    if not extra:
        return model
    bind = getattr(model, "bind", None)
    if callable(bind):
        try:
            return bind(extra_body=extra)
        except TypeError:
            try:
                return bind(model_kwargs={"extra_body": extra})
            except TypeError:
                pass
    return model
