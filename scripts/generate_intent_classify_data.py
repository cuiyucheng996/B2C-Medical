"""生成 C 端细意图训练语料（text,intent），意图名来自 data/intent_catalog.yaml。"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
B2C_SRC = B2C_ROOT / "src"
sys.path.insert(0, str(B2C_SRC))

from algorithm.intent_manifest import load_intent_manifest

DATA_DIR = B2C_ROOT / "data" / "intent_classify"
OUTPUT_CSV = DATA_DIR / "data.csv"
OTHER_CSV = DATA_DIR / "other_data.csv"

OTHER_SAMPLES = (
    "今天股市怎么样",
    "帮我写首诗",
    "附近有什么好吃的",
    "怎么刷短视频",
)


def generate_rows(*, per_class: int = 40, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    templates = load_intent_manifest().train_templates
    for intent, samples in templates.items():
        if not samples:
            continue
        for _ in range(per_class):
            rows.append((rng.choice(samples), intent))
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)


def main() -> None:
    rows = generate_rows()
    write_csv(OUTPUT_CSV, rows)
    write_csv(OTHER_CSV, [(text, "other") for text in OTHER_SAMPLES] * 10)
    print(f"写入 {OUTPUT_CSV} 共 {len(rows)} 条")
    print(f"写入 {OTHER_CSV}")


if __name__ == "__main__":
    main()
