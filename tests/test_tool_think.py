from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.tool_think import hint_from_verdict, parse_think_json, usable_envelopes
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ToolThinkParseTest(unittest.TestCase):
    def test_parse_fenced_json(self) -> None:
        row = parse_think_json('```json\n{"fit":false,"action":"retry_tools","reason":"不是这份化验"}\n```')
        self.assertFalse(row["fit"])
        self.assertEqual(row["action"], "retry_tools")

    def test_fit_forces_compose(self) -> None:
        row = parse_think_json('{"fit":true,"action":"retry_tools","reason":"x"}')
        self.assertTrue(row["fit"])
        self.assertEqual(row["action"], "compose")

    def test_usable_tails_only(self) -> None:
        messages = [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "t", "args": {}}]),
            ToolMessage(
                content='{"ok":true,"tool":"t","usable":true,"retryable":false,"error":"","summary":"s","data":{"a":1}}',
                tool_call_id="1",
                name="t",
            ),
        ]
        self.assertEqual(len(usable_envelopes(messages)), 1)

    def test_hint_retry_then_give_up(self) -> None:
        bad = {"fit": False, "action": "retry_tools", "reason": "对不上"}
        self.assertIn("再调一次", hint_from_verdict(bad, think_count=1))
        self.assertIn("停止调用", hint_from_verdict(bad, think_count=2))


if __name__ == "__main__":
    unittest.main()
