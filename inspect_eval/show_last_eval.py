"""不依赖浏览器：打印 logs 里最新 .eval 的输入与 C 端回复。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(LOG_DIR.glob("*.eval"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print(f"logs 目录无 .eval: {LOG_DIR}")
        sys.exit(1)
    path = files[0]
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path))
    print(f"=== {path.name} ===")
    print(f"task: {getattr(log.eval, 'task', '') or log.eval.task_id}")
    for sample in log.samples or []:
        inp = sample.input
        if isinstance(inp, str):
            text_in = inp
        else:
            text_in = json.dumps(inp, ensure_ascii=False)[:500]
        print(f"\n【输入】\n{text_in}")
        out = sample.output
        if out and out.completion:
            print(f"\n【C 端回复】\n{out.completion}")
        meta = sample.metadata or {}
        if meta:
            print(f"\n【metadata】\n{json.dumps(meta, ensure_ascii=False, indent=2)}")
    print("\n--- Inspect View ---")
    print("1. 保持 run_inspect_view.ps1 那个窗口不关")
    print("2. 浏览器打开: http://127.0.0.1:7575")
    print("3. 左侧点 eval 名称 → 再点 Sample 1")


if __name__ == "__main__":
    main()
