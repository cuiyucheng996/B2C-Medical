"""启动 B2C 陪聊 API（Graphiti 形态：session + agent + ingest/search）。

用法（本目录为项目根）：
    uv run python run.py
    uv run python run.py --port 8002
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("VECTOR_STORE_BACKEND", "chroma")

import uvicorn

B2C_ROOT = Path(__file__).resolve().parent
B2C_SRC = B2C_ROOT / "src"

if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))
if str(B2C_ROOT) not in sys.path:
    sys.path.insert(0, str(B2C_ROOT))

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 B2C 陪聊 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    os.environ.setdefault("B2C_SERVICE_NAME", "b2c-companion")
    os.environ["B2C_SERVICE_PORT"] = str(args.port)

    target = "api.server:app"
    print(f"Starting B2C companion -> {target} at http://{args.host}:{args.port}")
    extra: dict = {}
    if args.reload:
        extra = {
            "reload_dirs": [str(B2C_SRC)],
            "reload_includes": ["*.py", "*.html", "*.js", "*.css"],
        }
    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=str(B2C_SRC),
        **extra,
    )


if __name__ == "__main__":
    main()
