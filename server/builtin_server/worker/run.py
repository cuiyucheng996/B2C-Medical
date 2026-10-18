"""消费关场入库 / PDF 解析队列。

    uv run python server/builtin_server/worker/run.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
B2C_SRC = B2C_ROOT / "src"
for item in (str(B2C_ROOT), str(B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()

from server.builtin_server.jobs.worker import run_forever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    print("Starting heavy worker (Redis queue b2c:heavy:queue)")
    run_forever()


if __name__ == "__main__":
    main()
