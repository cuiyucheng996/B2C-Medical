"""Call run_kb_tool_probe same as eval UI POST /api/eval/kb/tools."""
import json
import sys
import time
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
B2C_SRC = B2C_ROOT / "src"
UVGRAPH_ROOT = B2C_ROOT.parent
for p in (str(UVGRAPH_ROOT), str(B2C_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

QUERY = "昨天晚上睡觉属实是太晚了 心脏突突跳动 现在很慌感觉"


def main() -> None:
    from algorithm.kb_eval import run_kb_tool_probe

    t0 = time.perf_counter()
    row = run_kb_tool_probe(
        query=QUERY,
        tools=["query_consult_graph"],
        handler="",
        rag_enhanced=None,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)
    timing = row.get("timing") or {}
    print("query:", QUERY)
    print("wall_ms", wall_ms)
    print("timing", json.dumps(timing, ensure_ascii=False))
    print("handler", row.get("handler"))
    print("entity_extracted", row.get("entity_extracted"))
    print("mentions", json.dumps(row.get("mentions"), ensure_ascii=False))
    for tr in row.get("tool_results") or []:
        print("tool", tr.get("tool"), "elapsed_ms", tr.get("elapsed_ms"), "ok", tr.get("ok"))


if __name__ == "__main__":
    main()
