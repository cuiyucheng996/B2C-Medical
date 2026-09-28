"""多意图归一化单测。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.intent_classify import CompanionIntentResult, normalize_intent_results


def _item(*, major: str, intent: str, handler: str) -> CompanionIntentResult:
    return CompanionIntentResult(
        intent=intent,
        major=major,
        confidence=0.8,
        source="test",
        reason="t",
        slots=(),
        handler=handler,
    )


class NormalizeIntentTest(unittest.TestCase):
    def test_drops_chat_when_real_intent(self) -> None:
        items = normalize_intent_results(
            [
                _item(major="闲聊", intent="寒暄打招呼", handler="chat_hello"),
                _item(major="公共知识问答", intent="查询某疾病的通俗解释", handler="kg_disease_plain"),
            ]
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].handler, "kg_disease_plain")

    def test_keeps_two_real_intents(self) -> None:
        items = normalize_intent_results(
            [
                _item(major="公共知识问答", intent="查询某疾病的通俗解释", handler="kg_disease_plain"),
                _item(major="具体任务", intent="提醒用户某日复查或随访", handler="task_remind"),
            ]
        )
        self.assertEqual([item.handler for item in items], ["kg_disease_plain", "task_remind"])

    def test_dedup_same_handler(self) -> None:
        items = normalize_intent_results(
            [
                _item(major="公共知识问答", intent="查询某疾病的通俗解释", handler="kg_disease_plain"),
                _item(major="公共知识问答", intent="查询某疾病的通俗解释", handler="kg_disease_plain"),
            ]
        )
        self.assertEqual(len(items), 1)

    def test_threshold_keeps_high_scores(self) -> None:
        low = CompanionIntentResult(
            intent="寒暄打招呼",
            major="闲聊",
            confidence=0.3,
            source="test",
            reason="t",
            slots=(),
            handler="chat_hello",
        )
        high = _item(
            major="具体任务",
            intent="提醒用户某日复查或随访",
            handler="task_remind",
        )
        high = CompanionIntentResult(
            intent=high.intent,
            major=high.major,
            confidence=0.9,
            source=high.source,
            reason=high.reason,
            slots=high.slots,
            handler=high.handler,
        )
        items = normalize_intent_results([low, high], min_confidence=0.55)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].handler, "task_remind")


if __name__ == "__main__":
    unittest.main()
