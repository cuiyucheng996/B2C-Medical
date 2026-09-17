"""C 端提示词：读 src/prompt/*.txt。工具说明仍写在各 @tool 的 docstring。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = ["PROMPT_DIR", "load_prompt"]

PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    filename = name if name.endswith(".txt") else f"{name}.txt"
    path = PROMPT_DIR / filename
    return path.read_text(encoding="utf-8").strip()
