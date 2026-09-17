from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from middleware.tool_pairs import drop_orphan_tool_messages


class DropOrphanToolMessagesTest(unittest.TestCase):
    def test_keeps_paired_tool(self) -> None:
        ai = AIMessage(content="", tool_calls=[{"id": "c1", "name": "t", "args": {}}])
        tool = ToolMessage(content="ok", tool_call_id="c1", name="t")
        human = HumanMessage(content="hi")
        out = drop_orphan_tool_messages([human, ai, tool])
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[-1], ToolMessage)

    def test_drops_orphan_tool(self) -> None:
        out = drop_orphan_tool_messages(
            [HumanMessage(content="hi"), ToolMessage(content="x", tool_call_id="gone", name="t")]
        )
        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0], HumanMessage)


if __name__ == "__main__":
    unittest.main()
