"""对话里随口提到的人、物、关系：规则抽取，写入 CompanionFact。

思路对齐 ConversationEntityMemory（实体+一句摘要）和 ConversationKGMemory（谁和谁做了什么），
不引入那两个类，也不另起存储。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["extract_talk_entities"]

_ROLES = (
    "爸爸",
    "妈妈",
    "父亲",
    "母亲",
    "老公",
    "老婆",
    "丈夫",
    "妻子",
    "儿子",
    "女儿",
    "孩子",
    "同事",
    "朋友",
    "室友",
    "老板",
    "领导",
    "邻居",
    "同学",
    "爸",
    "妈",
)

_SKIP_PERSON = frozenset(
    {
        "自己",
        "别人",
        "大家",
        "什么",
        "这个",
        "那个",
        "医院",
        "公司",
        "家里",
        "单位",
        "医生",
        "护士",
    }
)

_SKIP_THING = frozenset(
    {
        "药",
        "感冒药",
        "东西",
        "那个",
        "这个",
        "什么",
    }
)

_ROLE_ALIAS = {"爸": "爸爸", "妈": "妈妈"}
_ROLE_RE = re.compile(
    r"(?:我的?)?(?P<role>"
    + "|".join(sorted(_ROLES, key=len, reverse=True))
    + r")(?:[叫是](?P<name>[\u4e00-\u9fff]{1,4}))?"
)
_WITH_RES = (
    re.compile(
        r"(?:和|跟)(?P<who>[\u4e00-\u9fff]{2,4})一起(?P<act>吃饭|吵架|见面|出去|逛街|加班)?"
    ),
    re.compile(
        r"(?:和|跟)(?P<who>[\u4e00-\u9fff]{2,4})(?P<act>吃饭|吵架|见面|出去|逛街|加班)"
    ),
)
_THING_RE = re.compile(r"(?:买了|养了|用了|带了|丢了)(?P<thing>[\u4e00-\u9fff]{1,8})")


def extract_talk_entities(text: str) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    compact = raw.replace(" ", "")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: str, fact: str) -> None:
        span = (name or "").strip("的了啊呀呢吧 ")
        if not span or (kind, span) in seen:
            return
        seen.add((kind, span))
        rows.append(
            {
                "name": span,
                "fact": fact,
                "kind": kind,
                "negated": False,
            }
        )

    for match in _ROLE_RE.finditer(compact):
        role = match.group("role") or ""
        alias = (match.group("name") or "").strip()
        if alias and alias not in _SKIP_PERSON:
            add("person", alias, f"用户提到的人：{role}{alias}")
        add("person", _ROLE_ALIAS.get(role, role), f"用户提到的人：{role}")

    for pattern in _WITH_RES:
        for match in pattern.finditer(compact):
            who = (match.group("who") or "").strip()
            act = (match.group("act") or "").strip()
            if not who or who in _SKIP_PERSON or who in _ROLES or "一起" in who:
                continue
            add("person", who, f"用户提到的人：{who}")
            if act:
                add("event", f"{who}{act}", f"用户和{who}{act}")

    for match in _THING_RE.finditer(compact):
        thing = (match.group("thing") or "").strip("的了啊呀呢吧 ")
        if not thing or thing in _SKIP_THING or len(thing) > 8:
            continue
        add("thing", thing, f"用户提到的事物：{thing}")

    return rows
