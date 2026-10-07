"""进场长期记忆注入子图 worker 的裁剪。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage, HumanMessage

from unittest.mock import patch

from algorithm.memory_collect import (
    MEMORY_COLLECT_PREFIX,
    history_for_intent_worker,
    strip_memory_collect_messages,
)


class MemoryCollectWorkerTest(unittest.TestCase):
    def test_public_knowledge_omits_tree(self) -> None:
        tree = AIMessage(content=f"{MEMORY_COLLECT_PREFIX}\n头面部：3次\n" + "x" * 2000)
        hist = history_for_intent_worker(
            [tree, HumanMessage(content="你好")],
            handler="kg_symptom_care",
            major="公共知识问答",
        )
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0].content, "你好")

    def test_strip_removes_tree_from_history(self) -> None:
        tree = AIMessage(content=f"{MEMORY_COLLECT_PREFIX}\n头面部：3次")
        hist = strip_memory_collect_messages([tree, HumanMessage(content="你好")])
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0].content, "你好")

    def test_status_complaint_injects_fresh_summary(self) -> None:
        tree = AIMessage(content=f"{MEMORY_COLLECT_PREFIX}\n旧会话树\n" + "y" * 2000)
        composed = f"{MEMORY_COLLECT_PREFIX}\n头面部：3次\n头痛：4次\n" + "z" * 2000
        with patch("algorithm.memory_collect.compose_memory_collect", return_value=composed):
            hist = history_for_intent_worker(
                [tree, HumanMessage(content="最近头疼")],
                handler="status_complaint",
                major="近况陈述",
                user_id="u1",
            )
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[1].content, "最近头疼")
        text = str(hist[0].content)
        self.assertTrue(text.startswith(MEMORY_COLLECT_PREFIX))
        self.assertLessEqual(len(text), 400)
        self.assertIn("摘要", text)
        self.assertNotIn("旧会话树", text)


if __name__ == "__main__":
    unittest.main()
