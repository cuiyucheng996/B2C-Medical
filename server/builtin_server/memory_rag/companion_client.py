"""记忆页只调陪聊服务的内部接口，不直接 import 陪聊业务代码。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def _base() -> str:
    from infra.registry.discovery import companion_base_url

    return companion_base_url().rstrip("/")


def _token() -> str:
    return (os.getenv("B2C_INTERNAL_TOKEN") or "b2c-dev-internal").strip()


def companion_post(path: str, payload: dict[str, Any], *, timeout: float = 120) -> dict[str, Any]:
    from server.trace_http import internal_headers

    url = f"{_base()}{path}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=internal_headers(
            {
                "Content-Type": "application/json",
                "X-Internal-Token": _token(),
            }
        ),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return {"ok": False, "error": f"陪聊服务 {exc.code}：{detail}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"陪聊服务不可达（{_base()}）：{exc.reason}"}
    if not isinstance(body, dict):
        return {"ok": False, "error": "陪聊服务返回不是对象"}
    return body
