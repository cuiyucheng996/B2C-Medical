"""run_trace：trace_id 与 observability 载荷。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.run_trace import build_round_observability, start_turn_trace


class RunTraceTest(unittest.TestCase):
    def test_observability_from_langgraph(self) -> None:
        tid = start_turn_trace(session_id="s1", user_id="u1", query="hi")
        obs = build_round_observability(
            {
                "trace_id": tid,
                "engine": "langgraph",
                "pattern": "send",
                "workers": 1,
                "tool_runs": [
                    {
                        "major": "闲聊",
                        "intent": "寒暄",
                        "handler": "chat_hello",
                        "bound_tools": [],
                        "steps": [{"name": "x", "ok": "是", "source": "local", "summary": "ok"}],
                        "middleware_trace": [],
                    }
                ],
            }
        )
        self.assertEqual(obs["trace_id"], tid)
        self.assertEqual(len(obs["tool_runs"]), 1)
        self.assertEqual(obs["tool_runs"][0]["steps"][0]["name"], "x")


if __name__ == "__main__":
    unittest.main()
