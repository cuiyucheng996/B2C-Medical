"""交付检查：前端产物存在 + B/C 启动入口可编译。"""

from __future__ import annotations

import ast
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
ROOT = B2C_ROOT.parent
FRONT = B2C_ROOT / "web-frontend"
PY_FILES = [
    B2C_ROOT / "run.py",
    B2C_ROOT / "src" / "api" / "server.py",
    ROOT / "B2B business" / "run.py",
    ROOT / "B2B business" / "src" / "api" / "server.py",
]


def main() -> None:
    dist_html = FRONT / "dist" / "companion.html"
    if not dist_html.is_file():
        raise SystemExit(f"missing frontend build: {dist_html}")
    for path in PY_FILES:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        compile(source, str(path), "exec")
    print("deliver check ok")


if __name__ == "__main__":
    main()
