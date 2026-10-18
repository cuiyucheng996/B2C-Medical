"""启动定向采集专属服务。

在 B2C business 目录：
    uv run python server/builtin_server/directed_crawl/run.py
默认 http://127.0.0.1:8014

影刀教程中的 n8n 回调可改为：
    POST http://127.0.0.1:8014/webhook/start-crawl
    {"source":"amazon_rpa","status":"finished","job_type":"bsr","excel_path":"D:/n8ndata/amason-case/bsr_top50_YYYYMMDD.xlsx"}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

B2C_ROOT = Path(__file__).resolve().parents[3]
B2C_SRC = B2C_ROOT / "src"
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()


def main() -> None:
    parser = argparse.ArgumentParser(description="启动定向网页/RPA 采集服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8014)
    args = parser.parse_args()
    os.environ.setdefault("B2C_SERVICE_NAME", "b2c-directed-crawl")
    os.environ["B2C_SERVICE_PORT"] = str(args.port)
    print(f"Starting directed_crawl at http://{args.host}:{args.port}")
    uvicorn.run(
        "server.builtin_server.directed_crawl.app:app",
        host=args.host,
        port=args.port,
        app_dir=str(B2C_ROOT),
    )


if __name__ == "__main__":
    main()
