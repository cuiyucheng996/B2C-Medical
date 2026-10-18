"""通义联网搜索（百炼官方能力封装，供 MCP / 公共知识类意图）。"""

from __future__ import annotations

import logging
import os
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["dashscope_web_search"]


def _api_key() -> str:
    ensure_uvgraph_root()
    from src.configs import get_app_settings

    key = (get_app_settings().dashscope_api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")
    return key


def dashscope_web_search(query: str, *, model: str | None = None) -> dict[str, Any]:
    text = (query or "").strip()
    if not text:
        return {"success": False, "error": "query 为空", "answer": ""}
    ensure_uvgraph_root()
    from src.configs import get_yaml_config

    block = get_yaml_config().get("mcp") if isinstance(get_yaml_config().get("mcp"), dict) else {}
    aliyun = block.get("aliyun") if isinstance(block.get("aliyun"), dict) else {}
    model_name = (model or aliyun.get("search_model") or "qwen-plus").strip()
    proxy_keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")

    def _call_generation() -> Any:
        import dashscope
        from dashscope import Generation

        dashscope.api_key = _api_key()
        return Generation.call(
            model=model_name,
            messages=[{"role": "user", "content": text}],
            enable_search=True,
            result_format="message",
        )

    try:
        try:
            resp = _call_generation()
        except Exception as first_exc:
            err = str(first_exc)
            if "ProxyError" not in err and "9674" not in err and "proxy" not in err.lower():
                raise
            saved = {k: os.environ.pop(k) for k in proxy_keys if k in os.environ}
            try:
                resp = _call_generation()
            finally:
                for k, v in saved.items():
                    os.environ[k] = v
        if resp.status_code != 200:
            return {
                "success": False,
                "error": f"dashscope {resp.status_code} {getattr(resp, 'message', '')}",
                "answer": "",
            }
        choice = (resp.output or {}).get("choices") or []
        msg = choice[0].get("message", {}) if choice else {}
        content = str(msg.get("content") or "").strip()
        return {
            "success": True,
            "model": model_name,
            "query": text,
            "answer": content,
            "search_info": (resp.output or {}).get("search_info"),
        }
    except Exception as exc:
        logger.warning("dashscope web search failed", exc_info=True)
        return {"success": False, "error": str(exc)[:500], "answer": ""}
