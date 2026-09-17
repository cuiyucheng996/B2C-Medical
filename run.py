"""启动 B2C 陪聊 API（Graphiti 形态：session + agent + ingest/search）。

用法（项目根目录）：
    uv run python "B2C business/run.py"
    uv run python "B2C business/run.py" --port 8002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

B2C_ROOT = Path(__file__).resolve().parent
UVGRAPH_ROOT = B2C_ROOT.parent
B2C_SRC = B2C_ROOT / "src"

# uvgraph 根在前：`import src` 走主项目；B2C src 在后：`agent` / `session` / `api`
if str(UVGRAPH_ROOT) in sys.path:
    sys.path.remove(str(UVGRAPH_ROOT))
if str(B2C_SRC) in sys.path:
    sys.path.remove(str(B2C_SRC))
sys.path.insert(0, str(UVGRAPH_ROOT))
sys.path.insert(1, str(B2C_SRC))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 B2C 陪聊 API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

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
