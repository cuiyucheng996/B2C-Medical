"""B2C 路径：本目录 src + 内嵌 vendor/uvgraph（`import src`），不再依赖上级 uvgraph 根。"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
B2C_SRC = B2C_ROOT / "src"
VENDOR_UVGRAPH = B2C_ROOT / "vendor" / "uvgraph"
# 兼容旧脚本命名
UVGRAPH_ROOT = VENDOR_UVGRAPH


def ensure_b2c_runtime_path() -> Path:
    """B2C src 在前（algorithm/agent）；vendor 在后供 `import src.*`。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(B2C_ROOT / ".env")
    except ImportError:
        pass
    if str(B2C_ROOT) not in sys.path:
        sys.path.insert(0, str(B2C_ROOT))
    try:
        from infra.registry.nacos_config import apply_shared_config

        apply_shared_config()
    except Exception:
        pass
    vendor = str(VENDOR_UVGRAPH)
    local = str(B2C_SRC)
    for item in (vendor, local):
        if item in sys.path:
            sys.path.remove(item)
    sys.path.insert(0, local)
    sys.path.insert(1, vendor)
    return B2C_ROOT


def ensure_uvgraph_root() -> Path:
    """兼容旧调用：等价于 ensure_b2c_runtime_path。"""
    return ensure_b2c_runtime_path()


def ensure_b2c_src() -> Path:
    ensure_b2c_runtime_path()
    return B2C_SRC


def _algorithm_is_b2c() -> bool:
    pkg = sys.modules.get("algorithm")
    if pkg is None:
        return False
    path = str(getattr(pkg, "__file__", "") or "").replace("\\", "/")
    return "/B2C business/" in path or path.endswith("/B2C business/src/algorithm/__init__.py")


def _clear_algorithm_modules() -> None:
    for key in list(sys.modules):
        if key == "algorithm" or key.startswith("algorithm."):
            sys.modules.pop(key, None)


def import_b2c_algorithm(name: str):
    """C 端 algorithm 子模块。"""
    ensure_b2c_runtime_path()
    if not _algorithm_is_b2c():
        _clear_algorithm_modules()
    return importlib.import_module(f"algorithm.{name}")


@contextmanager
def b2b_src_imports() -> Iterator[None]:
    """已废弃：C 端业务请只用 ``b2c_b2b_bridge``，勿再切 sys.path 占包名。"""
    raise RuntimeError(
        "b2b_src_imports 已废弃。C 端请通过 b2c_b2b_bridge.b2b_*() 调用 B2B 能力，"
        "避免与 C 端 algorithm/schema 包名冲突。"
    )
    yield  # pragma: no cover
