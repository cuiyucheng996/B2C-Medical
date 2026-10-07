"""子 Agent system 静态/动态分块与 cache_control。"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage, HumanMessage

from algorithm.prefix_cache import (
    cache_mark_message,
    companion_layered_system_message,
    companion_split_system_message,
)
from middleware.context import CompanionRunContext
from prompt.companion_system import (
    build_companion_system_dynamic,
    build_companion_system_skill,
    build_companion_system_static,
)


class CompanionSystemCacheTest(unittest.TestCase):
    def test_static_same_across_handlers(self) -> None:
        a = build_companion_system_static()
        b = build_companion_system_static()
        self.assertEqual(a, b)
        self.assertIn("工具调度器", a)

    def test_dynamic_differs(self) -> None:
        d1 = build_companion_system_dynamic(major="公共知识问答", intent="a", handler="kg_exam_meaning")
        d2 = build_companion_system_dynamic(major="公共知识问答", intent="b", handler="kg_symptom_parts")
        self.assertNotEqual(d1, d2)

    def test_skill_same_handler_diff_intent(self) -> None:
        s1 = build_companion_system_skill(major="公共知识问答", handler="kg_handbook_general")
        s2 = build_companion_system_skill(major="公共知识问答", handler="kg_handbook_general")
        self.assertEqual(s1, s2)
        d1 = build_companion_system_dynamic(major="公共知识问答", intent="意图甲", handler="kg_handbook_general")
        d2 = build_companion_system_dynamic(major="公共知识问答", intent="意图乙", handler="kg_handbook_general")
        self.assertNotEqual(d1, d2)
        self.assertIn("意图甲", d1)
        self.assertIn("意图乙", d2)

    def test_split_message_blocks(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_PREFIX_CACHE": "1"}, clear=False):
            msg = companion_split_system_message("STATIC", "DYNAMIC")
        content = msg.content
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["text"], "STATIC")
        self.assertEqual(content[0].get("cache_control"), {"type": "ephemeral"})
        self.assertEqual(content[1]["text"], "DYNAMIC")
        self.assertNotIn("cache_control", content[1])

    def test_layered_message_blocks(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_PREFIX_CACHE": "1"}, clear=False):
            msg = companion_layered_system_message("STATIC", "SKILL", "TAIL")
        content = msg.content
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0].get("cache_control"), {"type": "ephemeral"})
        self.assertEqual(content[1]["text"], "SKILL")
        self.assertEqual(content[1].get("cache_control"), {"type": "ephemeral"})
        self.assertEqual(content[2]["text"], "TAIL")
        self.assertNotIn("cache_control", content[2])


    def test_cache_mark_human(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_PREFIX_CACHE": "1"}, clear=False):
            msg = cache_mark_message(HumanMessage(content="hello"))
        self.assertIsInstance(msg.content, list)
        self.assertEqual(msg.content[0].get("cache_control"), {"type": "ephemeral"})

    def test_prefix_anchor_on_context(self) -> None:
        ctx = CompanionRunContext(prefix_message_count=2)
        self.assertEqual(ctx.prefix_message_count, 2)


if __name__ == "__main__":
    unittest.main()
