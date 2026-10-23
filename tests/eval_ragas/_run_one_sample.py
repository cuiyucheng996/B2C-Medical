"""单次 RAGAS 样例（不经过 algorithm 包 __init__）。用法：uv run python tests/eval_ragas/_run_one_sample.py"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

algo = types.ModuleType("algorithm")
algo.__path__ = [str(SRC / "algorithm")]
sys.modules["algorithm"] = algo

for name in ("pdf_kb", "ragflow_kb", "knowledge_pipe", "ragas_pipeline"):
    importlib.import_module(f"algorithm.{name}")

ragas = sys.modules["algorithm.ragas_pipeline"]
knowledge = sys.modules["algorithm.knowledge_pipe"]

SOURCE = {
    "title": "头痛警示症状",
    "text": (
        "头痛若持续加重，或伴随呕吐、视力模糊、视物重影、意识改变，需尽快就医，"
        "排除脑出血、颅内压增高等严重病因；不要长期只靠止痛药硬扛。"
        "突发剧烈头痛（雷击样头痛）应立即急诊。"
    ),
}
QA = {
    "question": "头痛持续加重或伴呕吐视力变化该怎么办？",
    "ground_truth": "应尽快就医，排除严重病因，不要只靠止痛药硬扛，尤其突发剧烈头痛要急诊。",
}


def main() -> None:
    paths = ragas.default_paths()
    ragas.save_jsonl_text(paths["sources"], json.dumps(SOURCE, ensure_ascii=False))
    ragas.save_jsonl_text(paths["qa"], json.dumps(QA, ensure_ascii=False))
    ingest = ragas.ingest_rows(ragas.load_jsonl(paths["sources"]))
    print("=== 入库 ===")
    print(json.dumps(ingest, ensure_ascii=False, indent=2))
    print("\n=== 评测（local, top_k=5, 生成回答）===")
    try:
        out = ragas.run_ragas_eval(
            qa_path=paths["qa"],
            channel="local",
            top_k=5,
            generate_answer=True,
            save_report=True,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
