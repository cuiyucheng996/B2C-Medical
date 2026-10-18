"""周边服务把请求转给陪聊进程，避免本进程加载陪聊业务代码。"""

from __future__ import annotations

import json
import os
import uuid
import urllib.error
import urllib.request
from typing import Any


def companion_base() -> str:
    from infra.registry.discovery import companion_base_url

    return companion_base_url().rstrip("/")


def _token() -> str:
    return (os.getenv("B2C_INTERNAL_TOKEN") or "b2c-dev-internal").strip()


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {"ok": False, "error": text[:500]}


def request_json(method: str, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 180) -> tuple[int, Any]:
    from server.trace_http import internal_headers

    data = None
    headers = internal_headers({"X-Internal-Token": _token()})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{companion_base()}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except urllib.error.URLError as exc:
        return 503, {"ok": False, "error": f"陪聊服务不可达（{companion_base()}）：{exc.reason}"}


def request_multipart(
    path: str,
    *,
    file_field: str,
    filename: str,
    data: bytes,
    fields: dict[str, str],
    timeout: float = 300,
) -> tuple[int, Any]:
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode("utf-8")
        )
    safe_name = (filename or "upload.bin").replace('"', "")
    chunks.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{safe_name}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(data)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    from server.trace_http import internal_headers

    req = urllib.request.Request(
        f"{companion_base()}{path}",
        data=body,
        method="POST",
        headers=internal_headers(
            {
                "X-Internal-Token": _token(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            }
        ),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read())
    except urllib.error.URLError as exc:
        return 503, {"ok": False, "error": f"陪聊服务不可达（{companion_base()}）：{exc.reason}"}
