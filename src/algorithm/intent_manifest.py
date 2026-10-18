"""意图清单唯一数据源：middleware_third_party/intent_catalog.yaml。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from configs.data_paths import INTENT_CATALOG_PATH

_MANIFEST_PATH = INTENT_CATALOG_PATH

__all__ = [
    "intent_manifest_path",
    "load_intent_manifest",
    "validate_intent_manifest",
]


@dataclass(frozen=True)
class IntentManifest:
    major_order: tuple[str, ...]
    rule_order: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    by_name: dict[str, dict[str, Any]]
    hints: dict[str, str]
    rule_rules: tuple[tuple[str, tuple[str, ...]], ...]
    train_templates: dict[str, tuple[str, ...]]
    handlers: frozenset[str]


def intent_manifest_path() -> Path:
    return _MANIFEST_PATH


def _read_yaml() -> dict[str, Any]:
    import yaml

    path = _MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"意图清单缺失：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("intent_catalog.yaml 根节点必须是 mapping")
    return data


def validate_intent_manifest(data: dict[str, Any] | None = None) -> list[str]:
    """返回问题列表；空列表表示通过。"""
    raw = data if data is not None else _read_yaml()
    issues: list[str] = []
    intents = raw.get("intents")
    if not isinstance(intents, dict) or not intents:
        issues.append("intents 为空或不是 mapping")
        return issues
    rule_order = raw.get("rule_order")
    if not isinstance(rule_order, list):
        issues.append("rule_order 必须是 list")
        rule_order = []
    for name in rule_order:
        if name not in intents:
            issues.append(f"rule_order 引用未知意图：{name}")
    major_order = raw.get("major_order")
    if not isinstance(major_order, list) or not major_order:
        issues.append("major_order 不能为空")
    seen_handlers: set[str] = set()
    for name, row in intents.items():
        if not isinstance(row, dict):
            issues.append(f"意图 {name} 不是 mapping")
            continue
        if not str(row.get("major") or "").strip():
            issues.append(f"意图 {name} 缺少 major")
        handler = str(row.get("handler") or "").strip()
        if not handler:
            issues.append(f"意图 {name} 缺少 handler")
        elif handler in seen_handlers:
            issues.append(f"handler 重复：{handler}")
        else:
            seen_handlers.add(handler)
        kw = row.get("rule_keywords")
        if kw is not None and not isinstance(kw, list):
            issues.append(f"意图 {name} rule_keywords 必须是 list")
    return issues


@lru_cache(maxsize=1)
def load_intent_manifest() -> IntentManifest:
    raw = _read_yaml()
    problems = validate_intent_manifest(raw)
    if problems:
        raise ValueError("intent_catalog.yaml 校验失败：\n" + "\n".join(problems))

    intents: dict[str, Any] = raw["intents"]
    major_order = tuple(str(x) for x in raw["major_order"])
    rule_order = tuple(str(x) for x in raw["rule_order"])

    rows: list[dict[str, Any]] = []
    hints: dict[str, str] = {}
    train_templates: dict[str, tuple[str, ...]] = {}
    for name, row in intents.items():
        slots = tuple(str(s) for s in (row.get("slots") or []))
        item = {
            "intent": name,
            "major": str(row["major"]),
            "handler": str(row["handler"]),
            "slots": slots,
        }
        rows.append(item)
        hint = str(row.get("hint") or "").strip()
        if hint:
            hints[name] = hint
        tpl = row.get("train_templates")
        if isinstance(tpl, list) and tpl:
            train_templates[name] = tuple(str(t) for t in tpl)

    by_name = {row["intent"]: row for row in rows}
    rule_rules: list[tuple[str, tuple[str, ...]]] = []
    for name in rule_order:
        row = intents.get(name)
        if not isinstance(row, dict):
            continue
        kw = tuple(str(k) for k in (row.get("rule_keywords") or []) if str(k).strip())
        rule_rules.append((name, kw))

    extra = raw.get("extra_bind_handlers")
    extra_handlers = frozenset(str(h).strip() for h in (extra or []) if str(h).strip())
    handlers = frozenset(str(row["handler"]) for row in rows) | extra_handlers
    return IntentManifest(
        major_order=major_order,
        rule_order=rule_order,
        rows=tuple(rows),
        by_name=by_name,
        hints=hints,
        rule_rules=tuple(rule_rules),
        train_templates=train_templates,
        handlers=handlers,
    )
