"""业务冒烟：规则 → BERT（若已训）→ 打印大类/细意图/槽位/handler。"""

from __future__ import annotations

import sys
from pathlib import Path

B2C_SRC = Path(__file__).resolve().parents[1] / "src"
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))

from algorithm.intent_catalog import INTENT_BY_NAME
from algorithm.intent_classify import classify_companion_intent

SAMPLES = (
    "头部常见有哪些病",
    "胃炎有哪些症状",
    "血常规这个检查是查什么的",
    "转氨酶正常值是多少",
    "提醒我下周复查",
    "我说过哪里不舒服",
    "帮我看看这份化验单",
    "快救命",
    "我头疼好几天了",
    "你好",
    "今天股市怎么样",
)


def main() -> None:
    print(f"细意图共 {len(INTENT_BY_NAME)} 个")
    for text in SAMPLES:
        hit = classify_companion_intent(text)
        print(
            f"{text}\n  大类={hit.major} 细={hit.intent} 槽={list(hit.slots)} "
            f"handler={hit.handler} 来源={hit.source} {hit.recommend_action}\n"
        )


if __name__ == "__main__":
    main()
