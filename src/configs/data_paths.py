"""数据与评测产物路径（相对 B2C 项目根）。"""

from __future__ import annotations

from pathlib import Path

from configs.paths_b2c import B2C_ROOT

_DATA = B2C_ROOT / "data"
_MW = _DATA / "dev_resources" / "middleware_third_party"

INTENT_CATALOG_PATH = _DATA / "intent_catalog.yaml"
if not INTENT_CATALOG_PATH.is_file():
    INTENT_CATALOG_PATH = _MW / "intent_catalog.yaml"

UI_DRAFTS_DIR = _DATA / "ui_drafts"
DIRECTED_CRAWL_SOURCES = _DATA / "directed_crawl" / "sources.yaml"

INSPECT_EVAL_DIR = _DATA / "eval_inspect"
INSPECT_EVAL_LOG_DIR = INSPECT_EVAL_DIR / "logs"
INSPECT_EVAL_SAMPLES_PATH = INSPECT_EVAL_DIR / "agent_cases.jsonl"

__all__ = [
    "INTENT_CATALOG_PATH",
    "UI_DRAFTS_DIR",
    "DIRECTED_CRAWL_SOURCES",
    "INSPECT_EVAL_DIR",
    "INSPECT_EVAL_LOG_DIR",
    "INSPECT_EVAL_SAMPLES_PATH",
]
