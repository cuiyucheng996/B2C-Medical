"""启动文档入库专属服务。

在 B2C business 目录：
    uv run python server/builtin_server/doc_ingest/run.py
默认 http://127.0.0.1:8013
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

B2C_ROOT = Path(__file__).resolve().parents[2]
B2C_SRC = B2C_ROOT / "src"
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动文档入库专属服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    args = parser.parse_args()
    os.environ.setdefault("B2C_SERVICE_NAME", "b2c-doc-ingest")
    os.environ["B2C_SERVICE_PORT"] = str(args.port)
    print(f"Starting doc_ingest at http://{args.host}:{args.port}")
    uvicorn.run(
        "server.builtin_server.doc_ingest.app:app",
        host=args.host,
        port=args.port,
        app_dir=str(B2C_ROOT),
    )


if __name__ == "__main__":
    main()
