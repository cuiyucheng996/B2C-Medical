"""一次性灌入 Chroma：MySQL 入库 + 问诊对齐四表。

用法（B2C business 根目录）：
    uv run python scripts/bootstrap_chroma_vectors.py
    uv run python scripts/bootstrap_chroma_vectors.py --consult-only
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["UVGRAPH_EMBED_DEVICE"] = "cpu"

import argparse
import json
import logging
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
B2C_SRC = B2C_ROOT / "src"
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()

from src.algorithm.milvus_subprocess import vector_store_backend
from src.knowledge.consult_chroma_ingest import sync_consult_chroma_collections
from src.knowledge.mysql_chroma_ingest import ingest_mysql_to_chroma
from src.knowledge.vector_store_readiness import chroma_readiness_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ingest_table(table: str, *, reset: bool) -> dict:
    logger.info("MySQL → Chroma: table=%s reset=%s", table, reset)
    return ingest_mysql_to_chroma(table, reset=reset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus 同等数据灌入 Chroma")
    parser.add_argument("--consult-only", action="store_true")
    parser.add_argument("--skip-symptoms", action="store_true")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    reset = not args.no_reset

    backend = vector_store_backend()
    if backend != "chroma":
        raise SystemExit(f"vector_store.backend={backend}，请在 config.yaml 设为 chroma")

    before = chroma_readiness_report()
    logger.info("Chroma 灌库前: %s", before.get("detail"))

    report: dict = {"steps": []}
    if not args.consult_only:
        report["steps"].append(_ingest_table("disease", reset=reset))
        if not args.skip_symptoms:
            try:
                report["steps"].append(_ingest_table("symptoms", reset=reset))
            except Exception as exc:
                logger.warning("symptoms 表入库失败（可 --skip-symptoms）: %s", exc)
                report["symptoms_error"] = str(exc)

    report["consult"] = sync_consult_chroma_collections(reset=reset)
    after = chroma_readiness_report()
    report["readiness_after"] = after

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not after.get("ready"):
        missing = after.get("missing") or []
        raise SystemExit(f"仍缺 collection: {', '.join(missing)}")
    logger.info("Chroma 向量库 collection 齐全。")


if __name__ == "__main__":
    main()
