"""把各工具原始返回收成 CompanionToolEnvelope JSON。"""

from __future__ import annotations

import json
from typing import Any

from schema.tool_result import CompanionToolEnvelope

__all__ = ["dump_envelope", "is_envelope", "normalize_tool_content", "substance_of"]

_MAX_CHARS = 8000


def substance_of(data: dict[str, Any]) -> bool:
    if data.get("items") or data.get("facts") or data.get("extracts"):
        return True
    if data.get("current_disease") or data.get("current_body_part"):
        return True
    if data.get("current_symptoms"):
        return True
    if data.get("policy") or data.get("note"):
        return True
    if str(data.get("summary") or "").strip():
        return True
    return False


def is_envelope(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return "ok" in payload and "usable" in payload and "data" in payload and "tool" in payload


def dump_envelope(
    *,
    tool: str,
    ok: bool = True,
    error: str = "",
    summary: str = "",
    data: dict[str, Any] | None = None,
    usable: bool | None = None,
    retryable: bool | None = None,
) -> str:
    body = dict(data or {})
    if usable is None:
        usable = bool(ok) and substance_of(body)
    if retryable is None:
        # 具体任务类（note_task_*）重调会重复发提醒，一律不允许重试
        retryable = (not usable) and not (tool.startswith("note_task_") or tool == "safety_policy_hint")
    env = CompanionToolEnvelope(
        ok=bool(ok),
        tool=tool,
        usable=bool(usable),
        retryable=bool(retryable),
        error=(error or "")[:400],
        summary=(summary or "")[:400],
        data=body,
    )
    payload = env.model_dump()
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > _MAX_CHARS:
        env.data = {"truncated": True, "preview": text[: _MAX_CHARS - 80]}
        env.summary = (env.summary or "结果过长已截断")[:400]
        text = json.dumps(env.model_dump(), ensure_ascii=False)
    return text


def _parse_raw(content: Any) -> Any:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        return content
    text = str(content or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_tool_content(tool: str, content: Any, *, status: str = "") -> str:
    parsed = _parse_raw(content)
    if is_envelope(parsed):
        return json.dumps(parsed, ensure_ascii=False)
    if status == "error":
        return dump_envelope(
            tool=tool,
            ok=False,
            error=str(content or "tool error")[:400],
            retryable=True,
            usable=False,
            data={},
        )
    if isinstance(parsed, dict):
        ok = parsed.get("ok")
        if ok is False:
            return dump_envelope(
                tool=tool,
                ok=False,
                error=str(parsed.get("reason") or parsed.get("error") or "失败"),
                summary=str(parsed.get("summary") or ""),
                data={k: v for k, v in parsed.items() if k not in {"ok"}},
                usable=False,
                retryable=True,
            )
        data = {k: v for k, v in parsed.items() if k != "ok"}
        summary = str(parsed.get("summary") or parsed.get("note") or parsed.get("policy") or "")
        return dump_envelope(tool=tool, ok=True, summary=summary, data=data)
    if isinstance(parsed, list):
        return dump_envelope(
            tool=tool,
            ok=True,
            summary=f"{len(parsed)} 条",
            data={"facts": parsed} if parsed else {"facts": []},
            usable=bool(parsed),
        )
    if parsed is None or parsed == "（暂无仍有效的长期事实）":
        return dump_envelope(
            tool=tool,
            ok=True,
            summary="暂无命中",
            data={},
            usable=False,
            retryable=True,
        )
    return dump_envelope(
        tool=tool,
        ok=True,
        summary=str(parsed)[:200],
        data={"raw": str(parsed)[:2000]},
        usable=bool(str(parsed).strip()),
        retryable=not bool(str(parsed).strip()),
    )
