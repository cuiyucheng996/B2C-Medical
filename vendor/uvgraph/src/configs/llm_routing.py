"""LLM 提供商解析与代理策略（国内直连 / 海外走代理）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from .config_loader import AppSettings, app_settings

LlmProvider = Literal["deepseek", "kimi", "openai"]
LlmProxyMode = Literal["auto", "always", "never"]

# 已知国内 API 域名片段（base_url 命中则默认不走代理）
_DOMESTIC_HOST_MARKERS: tuple[str, ...] = (
    ".cn",
    "deepseek.com",
    "moonshot.cn",
    "aliyuncs.com",
    "dashscope",
    "volces.com",
    "bigmodel.cn",
    "localhost",
    "127.0.0.1",
)

# 已知海外默认域名（无自定义 base_url 时判定）
_OVERSEAS_DEFAULT_HOSTS: tuple[str, ...] = (
    "api.openai.com",
    "openai.azure.com",
    "anthropic.com",
)

__all__ = [
    "ActiveLlmConfig",
    "LlmProvider",
    "LlmProxyMode",
    "httpx_kwargs_for_llm",
    "infer_needs_proxy",
    "is_domestic_endpoint",
    "list_configured_llm_providers",
    "resolve_active_llm_provider",
    "resolve_llm_proxy_mode",
]


@dataclass(frozen=True, slots=True)
class ActiveLlmConfig:
    """当前生效的聊天模型配置（与 ``get_chat_model`` 优先级一致）。"""

    provider: LlmProvider
    api_key: str
    base_url: str
    model: str
    needs_proxy: bool


def resolve_llm_proxy_mode() -> LlmProxyMode:
    raw = (os.environ.get("LLM_PROXY_MODE") or "auto").strip().lower()
    if raw in ("always", "never"):
        return raw  # type: ignore[return-value]
    return "auto"


def is_domestic_endpoint(base_url: str | None) -> bool:
    """根据 base_url 主机名粗判是否国内/可直连端点。"""
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return False
    if host.endswith(".cn"):
        return True
    return any(marker in host for marker in _DOMESTIC_HOST_MARKERS)


def infer_needs_proxy(provider: LlmProvider, base_url: str | None) -> bool:
    """推断该提供商在当前 base_url 下是否需要 HTTP 代理。"""
    if is_domestic_endpoint(base_url):
        return False

    if provider in ("deepseek", "kimi"):
        return False

    if provider == "openai":
        host = (urlparse(base_url or "").hostname or "").lower()
        if not host:
            return True
        if any(marker in host for marker in _OVERSEAS_DEFAULT_HOSTS):
            return True
        return not is_domestic_endpoint(base_url)

    return False


def _effective_needs_proxy(provider: LlmProvider, base_url: str | None) -> bool:
    mode = resolve_llm_proxy_mode()
    if mode == "always":
        return True
    if mode == "never":
        return False
    return infer_needs_proxy(provider, base_url)


def resolve_active_llm_provider(
    settings: AppSettings | None = None,
) -> ActiveLlmConfig | None:
    """解析当前将使用的 LLM（DeepSeek → Kimi → OpenAI 优先级）。"""
    cfg = settings or app_settings

    if cfg.deepseek_api_key:
        base_url = cfg.deepseek_base_url
        return ActiveLlmConfig(
            provider="deepseek",
            api_key=cfg.deepseek_api_key,
            base_url=base_url,
            model=cfg.deepseek_model,
            needs_proxy=_effective_needs_proxy("deepseek", base_url),
        )

    if cfg.kimi_api_key:
        base_url = cfg.kimi_base_url
        return ActiveLlmConfig(
            provider="kimi",
            api_key=cfg.kimi_api_key,
            base_url=base_url,
            model=cfg.kimi_model,
            needs_proxy=_effective_needs_proxy("kimi", base_url),
        )

    if cfg.openai_api_key:
        base_url = cfg.openai_base_url or "https://api.openai.com/v1"
        return ActiveLlmConfig(
            provider="openai",
            api_key=cfg.openai_api_key,
            base_url=base_url,
            model=cfg.openai_model,
            needs_proxy=_effective_needs_proxy("openai", base_url),
        )

    return None


def list_configured_llm_providers(
    settings: AppSettings | None = None,
) -> list[ActiveLlmConfig]:
    """列出已配置 Key 的全部提供商（供后续多模型路由扩展）。"""
    cfg = settings or app_settings
    providers: list[ActiveLlmConfig] = []

    if cfg.deepseek_api_key:
        base_url = cfg.deepseek_base_url
        providers.append(
            ActiveLlmConfig(
                provider="deepseek",
                api_key=cfg.deepseek_api_key,
                base_url=base_url,
                model=cfg.deepseek_model,
                needs_proxy=_effective_needs_proxy("deepseek", base_url),
            )
        )
    if cfg.kimi_api_key:
        base_url = cfg.kimi_base_url
        providers.append(
            ActiveLlmConfig(
                provider="kimi",
                api_key=cfg.kimi_api_key,
                base_url=base_url,
                model=cfg.kimi_model,
                needs_proxy=_effective_needs_proxy("kimi", base_url),
            )
        )
    if cfg.openai_api_key:
        base_url = cfg.openai_base_url or "https://api.openai.com/v1"
        providers.append(
            ActiveLlmConfig(
                provider="openai",
                api_key=cfg.openai_api_key,
                base_url=base_url,
                model=cfg.openai_model,
                needs_proxy=_effective_needs_proxy("openai", base_url),
            )
        )
    return providers


def httpx_kwargs_for_llm(active: ActiveLlmConfig) -> dict[str, Any]:
    """构造 LLM 专用 httpx 参数：``trust_env=False``，按提供商决定是否走代理。"""
    from .proxy import read_proxy_url_from_env

    kwargs: dict[str, Any] = {"trust_env": False, "timeout": 120.0}
    if active.needs_proxy:
        proxy = read_proxy_url_from_env()
        if proxy:
            kwargs["proxy"] = proxy
    return kwargs
