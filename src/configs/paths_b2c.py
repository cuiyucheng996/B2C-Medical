"""B2C 路径：本目录 src + uvgraph 根。"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
B2C_SRC = B2C_ROOT / "src"
UVGRAPH_ROOT = B2C_ROOT.parent
B2B_SRC = UVGRAPH_ROOT / "B2B business" / "src"

_B2B_TOP = ("algorithm", "knowledge", "schema", "configs")
_b2b_loaded: dict[str, object] = {}


def ensure_uvgraph_root() -> Path:
    """`src.*` 必须指向 uvgraph 根下的包，不能被 B2C 自己的 src 目录抢先。"""
    root = str(UVGRAPH_ROOT)
    local = str(B2C_SRC)
    for item in (root, local):
        if item in sys.path:
            sys.path.remove(item)
    sys.path.insert(0, root)
    sys.path.insert(1, local)
    return UVGRAPH_ROOT


def ensure_b2c_src() -> Path:
    ensure_uvgraph_root()
    return B2C_SRC


def _is_b2b_top(name: str) -> bool:
    return name in _B2B_TOP or any(name.startswith(f"{item}.") for item in _B2B_TOP)


@contextmanager
def b2b_src_imports() -> Iterator[None]:
    """临时切到 B2B src，复用意图/UIE/否定/对齐。

    C 端自己也有 ``algorithm`` / ``schema`` / ``configs``，必须换栈再 import。
    已加载的 B2B 模块（UIE、意图 BERT）缓存在 ``_b2b_loaded``，避免每轮对话重装模型。
    不把四轮问诊图搬进 C 端。
    """
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if _is_b2b_top(key):
            saved[key] = sys.modules.pop(key)
    sys.modules.update(_b2b_loaded)
    b2b = str(B2B_SRC)
    if b2b in sys.path:
        sys.path.remove(b2b)
    sys.path.insert(0, b2b)
    try:
        yield
        for key in list(sys.modules):
            if _is_b2b_top(key):
                _b2b_loaded[key] = sys.modules[key]
    finally:
        for key in list(sys.modules):
            if _is_b2b_top(key):
                sys.modules.pop(key, None)
        if b2b in sys.path:
            sys.path.remove(b2b)
        sys.modules.update(saved)
        ensure_uvgraph_root()
