"""公共知识问答：同一大类保留多条达阈值细意图。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.intent_classify import (
    _merge_fine_by_handler,
    _pack,
    _reconcile_intents_with_major_scores,
)


class KnowledgeMultiFineTest(unittest.TestCase):
    def test_merge_keeps_distinct_handlers(self) -> None:
        a = _pack(
            major="公共知识问答",
            intent="查询某部位有哪些常见疾病",
            confidence=0.8,
            source="t",
            reason="a",
        )
        b = _pack(
            major="公共知识问答",
            intent="查询某疾病的通俗解释",
            confidence=0.76,
            source="t",
            reason="b",
        )
        merged = _merge_fine_by_handler([a, b, a])
        self.assertEqual(len(merged), 2)
        self.assertEqual({m.handler for m in merged}, {"kg_part_diseases", "kg_disease_plain"})

    def test_reconcile_keeps_two_kg_from_llm_items(self) -> None:
        packed = [
            _pack(
                major="公共知识问答",
                intent="查询某部位有哪些常见疾病",
                confidence=0.82,
                source="llm",
                reason="部位",
            ),
            _pack(
                major="公共知识问答",
                intent="查询某疾病的通俗解释",
                confidence=0.78,
                source="llm",
                reason="白话",
            ),
        ]
        scores = {name: 0.1 for name in ("公共知识问答", "具体任务", "闲聊", "other")}
        scores["公共知识问答"] = 0.88
        with patch(
            "algorithm.intent_classify.classify_fine_intents_for_major",
            return_value=[],
        ):
            rows = _reconcile_intents_with_major_scores(packed, scores, "头部有啥病，脑卒中是不是中风")
        kg = [r for r in rows if r.major == "公共知识问答"]
        self.assertEqual(len(kg), 2)
        self.assertEqual({r.handler for r in kg}, {"kg_part_diseases", "kg_disease_plain"})


if __name__ == "__main__":
    unittest.main()
