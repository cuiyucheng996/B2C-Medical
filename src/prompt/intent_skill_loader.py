"""细分意图 Skill（SOP）唯一运行时源：src/prompt/intent_skills/{handler}.md。"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from prompt import PROMPT_DIR

__all__ = [
    "INTENT_SKILLS_DIR",
    "intent_skill_path",
    "load_intent_skill",
    "load_major_skill",
    "list_intent_skill_handlers",
]

INTENT_SKILLS_DIR = PROMPT_DIR / "intent_skills"
_MAJOR_DIR = INTENT_SKILLS_DIR / "_major"
_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_INCLUDE = re.compile(r"\{\{include:([^}]+)\}\}")


def intent_skill_path(handler: str) -> Path:
    return INTENT_SKILLS_DIR / f"{(handler or '').strip()}.md"


def _strip_frontmatter(text: str) -> str:
    body = _FRONTMATTER.sub("", text.strip(), count=1)
    return body.strip()


def _expand_includes(text: str, *, depth: int = 0) -> str:
    if depth > 5:
        return text

    def _repl(match: re.Match[str]) -> str:
        rel = match.group(1).strip().lstrip("/").replace("\\", "/")
        path = (INTENT_SKILLS_DIR / rel).resolve()
        if not path.is_file() or INTENT_SKILLS_DIR not in path.parents:
            return match.group(0)
        inner = _strip_frontmatter(path.read_text(encoding="utf-8"))
        return _expand_includes(inner, depth=depth + 1)

    return _INCLUDE.sub(_repl, text)


@lru_cache(maxsize=64)
def load_intent_skill(handler: str) -> str | None:
    h = (handler or "").strip()
    if not h:
        return None
    path = intent_skill_path(h)
    if not path.is_file():
        return None
    return _expand_includes(_strip_frontmatter(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=16)
def load_major_skill(major: str) -> str | None:
    m = (major or "").strip()
    if not m:
        return None
    path = _MAJOR_DIR / f"{m}.md"
    if not path.is_file():
        return None
    return _expand_includes(_strip_frontmatter(path.read_text(encoding="utf-8")))


def list_intent_skill_handlers() -> tuple[str, ...]:
    if not INTENT_SKILLS_DIR.is_dir():
        return ()
    names: list[str] = []
    for path in sorted(INTENT_SKILLS_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        names.append(path.stem)
    return tuple(names)
