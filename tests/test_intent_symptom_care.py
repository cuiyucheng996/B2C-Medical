"""症状护理/止血应落到「查询某症状的生活注意事项」。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.intent_classify import classify_fine_for_major
from algorithm.intent_rule import match_intent, match_major_intent
from tools.setup import INTENT_BIND


class SymptomCareIntentTest(unittest.TestCase):
    QUERY = "发生创伤出血时 怎么止血"

    def test_rule_hits_care(self) -> None:
        self.assertEqual(
            match_intent(self.QUERY, major="公共知识问答"),
            "查询某症状的生活注意事项",
        )

    def test_major_not_emergency(self) -> None:
        self.assertEqual(match_major_intent(self.QUERY), "公共知识问答")

    def test_bind_handbook(self) -> None:
        self.assertEqual(
            INTENT_BIND[("公共知识问答", "kg_symptom_care")],
            ("search_exam_handbook",),
        )

    def test_fine_overrides_wrong_llm(self) -> None:
        fake = mock.Mock()
        fake.model_copy.return_value = fake
        fake.invoke.return_value = mock.Mock(
            content='{"intent":"未细分","reason":"像急救"}'
        )
        row = classify_fine_for_major(self.QUERY, major="公共知识问答", model=fake, llm_seconds=2)
        self.assertEqual(row.intent, "查询某症状的生活注意事项")
        self.assertEqual(row.handler, "kg_symptom_care")


if __name__ == "__main__":
    unittest.main()
