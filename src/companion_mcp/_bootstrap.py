"""MCP stdio 子进程：与 run.py 一致的 sys.path（B2C src + vendor）。"""

from __future__ import annotations

import sys
from pathlib import Path

B2C_SRC = Path(__file__).resolve().parents[1]
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()
