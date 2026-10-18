"""实现已迁到 server/builtin_server/report_read/ingest.py。"""

from __future__ import annotations

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from server.builtin_server.report_read import ingest as _ingest

globals().update({key: value for key, value in vars(_ingest).items() if not key.startswith("__")})
__all__ = list(getattr(_ingest, "__all__", []))
