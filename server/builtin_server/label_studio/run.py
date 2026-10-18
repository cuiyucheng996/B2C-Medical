"""启动 Label Studio 标注服务。

在 B2C business 目录：
    uv run python server/builtin_server/label_studio/run.py
默认 http://127.0.0.1:8080
数据目录 model/labeling/label_studio
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = B2C_ROOT / "model" / "labeling" / "label_studio"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Label Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["LABEL_STUDIO_BASE_DATA_DIR"] = str(DATA_DIR)
    os.environ["DJANGO_SETTINGS_MODULE"] = "label_studio.core.settings.label_studio"
    sys.argv = [
        "label-studio",
        "start",
        "--no-browser",
        "--internal-host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"Starting Label Studio at http://{args.host}:{args.port}")
    from label_studio.server import main as serve

    raise SystemExit(serve())


if __name__ == "__main__":
    main()
