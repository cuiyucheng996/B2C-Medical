"""具体任务推到 n8n：提醒建日程、打卡记完成。未配置 URL 时跳过。"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = ["push_companion_task"]


def push_companion_task(
    *,
    kind: str,
    user_id: str = "",
    session_id: str = "",
    title: str = "",
    when_text: str = "",
    query: str = "",
    start_iso: str = "",
) -> dict[str, Any]:
    """POST JSON 到 n8n Webhook。返回 {ok, skipped?, error?, status?}。"""
    ensure_uvgraph_root()
    from src.configs import get_app_settings, get_yaml_config

    settings = get_app_settings()
    cfg = get_yaml_config().get("n8n") if isinstance(get_yaml_config().get("n8n"), dict) else {}
    if cfg.get("enabled") is False:
        return {"ok": True, "skipped": "n8n_disabled"}
    url = (getattr(settings, "n8n_webhook_url", None) or "").strip() or str(cfg.get("webhook_url") or "").strip()
    if not url:
        return {"ok": True, "skipped": "n8n_not_configured"}
    timeout = float(cfg.get("timeout_seconds") or 8)
    payload = {
        "kind": (kind or "").strip() or "remind",
        "user_id": (user_id or "").strip(),
        "session_id": (session_id or "").strip(),
        "title": (title or "").strip(),
        "when_text": (when_text or "").strip(),
        "query": (query or "").strip(),
        "start_iso": (start_iso or "").strip(),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    secret = (getattr(settings, "n8n_webhook_secret", None) or "").strip()
    if secret:
        headers["X-Webhook-Secret"] = secret
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": int(getattr(resp, "status", 200) or 200), "body": raw[:500]}
    except urllib.error.HTTPError as exc:
        logger.warning("n8n webhook http %s", exc.code)
        return {"ok": False, "error": f"http_{exc.code}", "retryable": True}
    except Exception as exc:
        logger.warning("n8n webhook failed: %s", exc)
        return {"ok": False, "error": str(exc)[:200], "retryable": True}
