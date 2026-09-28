"""按 user_id 持久化 C 端 UI 草稿（session 释放后 / 跨设备兜底）。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["load_user_ui_draft", "save_user_ui_draft", "merge_user_ui_draft"]

_DRAFT_ROOT = Path(__file__).resolve().parents[2] / "data" / "ui_drafts"
_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _path(user_id: str) -> Path:
    uid = (user_id or "").strip()
    if not uid:
        raise ValueError("user_id 不能为空")
    safe = _SAFE.sub("_", uid)[:120]
    return _DRAFT_ROOT / f"{safe}.json"


def load_user_ui_draft(user_id: str) -> dict[str, Any]:
    path = _path(user_id)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def merge_user_ui_draft(user_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    base = load_user_ui_draft(user_id)
    for key, val in patch.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])
            merged.update(val)
            base[key] = merged
        else:
            base[key] = val
    base["updated_at"] = datetime.now(timezone.utc).isoformat()
    base["user_id"] = (user_id or "").strip()
    save_user_ui_draft(user_id, base)
    return base


def save_user_ui_draft(user_id: str, payload: dict[str, Any]) -> None:
    path = _path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["user_id"] = (user_id or "").strip()
    body["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
