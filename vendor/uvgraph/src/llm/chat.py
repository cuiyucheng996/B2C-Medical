"""项目内 LLM 初始化（DeepSeek / Kimi / OpenAI 兼容）。"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from ..configs import app_settings
from ..configs.llm_routing import httpx_kwargs_for_llm, resolve_active_llm_provider

logger = logging.getLogger(__name__)

__all__ = ["get_chat_model", "resolve_active_llm_provider"]


@lru_cache
def get_chat_model() -> BaseChatModel | None:
    active = resolve_active_llm_provider(app_settings)
    if active is None:
        return None

    client_kwargs = httpx_kwargs_for_llm(active)
    http_client = httpx.Client(**client_kwargs)
    http_async_client = httpx.AsyncClient(**client_kwargs)

    model_kwargs: dict[str, Any] = {
        "api_key": active.api_key,
        "base_url": active.base_url,
        "temperature": 0,
        "http_client": http_client,
        "http_async_client": http_async_client,
    }

    logger.info(
        "chat model provider=%s model=%s base_url=%s proxy=%s mode=%s",
        active.provider,
        active.model,
        active.base_url,
        client_kwargs.get("proxy") or "direct",
        "on" if active.needs_proxy else "off",
    )

    return init_chat_model(f"openai:{active.model}", **model_kwargs)
