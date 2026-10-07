"""问诊 API 接入层与入口通道配置（读 config.yaml + .env 密钥）。"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .config_loader import get_yaml_config

__all__ = ["get_medical_api_config"]


def _section_dict(data: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return {}
    return {k: v for k, v in section_data.items() if v is not None}


@lru_cache
def get_medical_api_config() -> dict[str, Any]:
    """返回问诊 API 鉴权、限流与输入管道配置。"""
    cfg = _section_dict(get_yaml_config(), "medical_api")
    rate = cfg.get("rate_limit") if isinstance(cfg.get("rate_limit"), dict) else {}
    input_cfg = cfg.get("input") if isinstance(cfg.get("input"), dict) else {}
    asr_cfg = cfg.get("asr_normalize") if isinstance(cfg.get("asr_normalize"), dict) else {}
    memory_cfg = cfg.get("user_memory") if isinstance(cfg.get("user_memory"), dict) else {}

    default_channel = str(input_cfg.get("default_channel", "text")).strip().lower()
    if default_channel not in ("text", "asr"):
        default_channel = "text"

    return {
        "api_key_required": bool(cfg.get("api_key_required", False)),
        "api_key": os.getenv("MEDICAL_API_KEY", "").strip(),
        "rate_limit_enabled": bool(rate.get("enabled", True)),
        "rate_limit_max_requests": int(rate.get("max_requests", 60)),
        "rate_limit_window_seconds": int(rate.get("window_seconds", 60)),
        "max_text_length": int(input_cfg.get("max_text_length", 2000)),
        "default_input_channel": default_channel,
        "asr_collapse_spaces": bool(asr_cfg.get("collapse_spaces", True)),
        "asr_strip_punctuation_noise": bool(asr_cfg.get("strip_punctuation_noise", True)),
        "user_memory_enabled": bool(memory_cfg.get("enabled", True)),
    }
