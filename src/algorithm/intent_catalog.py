"""C 端意图清单：大类 + 细意图（权威源 data/intent_catalog.yaml）。"""

from __future__ import annotations

from typing import Any, Literal

from algorithm.intent_manifest import load_intent_manifest

MajorIntent = Literal[
    "公共知识问答",
    "具体任务",
    "个人记忆问答",
    "报告指标解读",
    "风险拦截",
    "近况陈述",
    "闲聊",
    "other",
]

_manifest = load_intent_manifest()
INTENT_ROWS: tuple[dict[str, Any], ...] = _manifest.rows
INTENT_BY_NAME: dict[str, dict[str, Any]] = _manifest.by_name
INTENT_HINTS: dict[str, str] = _manifest.hints

__all__ = ["INTENT_BY_NAME", "INTENT_HINTS", "INTENT_ROWS", "MajorIntent", "major_of", "intent_prompt_lines"]


def intent_prompt_lines(*, major: str | None = None) -> str:
    lines: list[str] = []
    for row in INTENT_ROWS:
        name = str(row["intent"])
        if major and str(row.get("major") or "") != major:
            continue
        hint = INTENT_HINTS.get(name, "")
        extra = f"：{hint}" if hint else ""
        if major:
            lines.append(f"- {name}{extra}")
        else:
            lines.append(f"- {name}（大类：{row['major']}）{extra}")
    return "\n".join(lines)


def major_of(intent: str) -> str:
    row = INTENT_BY_NAME.get(intent or "")
    if row:
        return str(row["major"])
    if intent in ("other", "未知意图", ""):
        return "other"
    return "other"
