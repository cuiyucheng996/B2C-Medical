"""泛化健康科普细意图规则。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.intent_rule import match_intent, match_major_intent
from tools.companion_tools import tools_for_intent
from tools.registry import reset_companion_registry
from tools.setup import register_all
from tools.registry import CompanionToolRegistry


class HandbookGeneralIntentTest(unittest.TestCase):
    def test_rule_match_blood_donation(self) -> None:
        self.assertEqual(match_major_intent("无偿献血有什么好处"), "公共知识问答")
        self.assertEqual(
            match_intent("无偿献血有什么好处", major="公共知识问答"),
            "查询一般健康科普或常见问题",
        )

    def test_tool_bind(self) -> None:
        reset_companion_registry()
        reg = CompanionToolRegistry()
        register_all(reg)
        names = [t.name for t in tools_for_intent(major="公共知识问答", handler="kg_handbook_general")]
        self.assertEqual(names, ["search_exam_handbook", "search_chroma_knowledge"])


if __name__ == "__main__":
    unittest.main()
