"""工具 HITL 决策解析。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.companion_hitl import (
    HITL_KIND_TOOL,
    build_tool_hitl_resume,
    format_tool_hitl_question,
    parse_client_hitl_decision,
)


class CompanionToolHitlTest(unittest.TestCase):
    def test_format_question(self) -> None:
        q = format_tool_hitl_question(
            {
                "action_requests": [
                    {"name": "note_task_remind", "args": {"when": "明天"}, "description": "设提醒"},
                ]
            }
        )
        self.assertIn("note_task_remind", q)
        self.assertIn("明天", q)

    def test_parse_approve(self) -> None:
        rows = parse_client_hitl_decision({"type": "approve"}, count=1)
        self.assertEqual(rows[0]["type"], "approve")

    def test_build_resume_reject(self) -> None:
        pending = {
            "action_requests": [{"name": "note_task_remind", "args": {}}],
            "hitl_kind": HITL_KIND_TOOL,
        }
        payload = build_tool_hitl_resume(
            pending, decision={"type": "reject"}, user_text="先不要"
        )
        self.assertEqual(payload["decisions"][0]["type"], "reject")


if __name__ == "__main__":
    unittest.main()
