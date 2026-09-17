"""中间件栈相关单测。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from middleware import companion_middleware, spell_check_middleware
from middleware.mcp_guard import is_denied_mcp_tool


class MiddlewareStackTest(unittest.TestCase):
    def test_companion_stack_not_empty(self) -> None:
        self.assertGreaterEqual(len(companion_middleware()), 6)

    def test_spell_stack_not_empty(self) -> None:
        self.assertGreaterEqual(len(spell_check_middleware()), 3)

    def test_mcp_deny(self) -> None:
        self.assertTrue(is_denied_mcp_tool("graphiti_add_memory"))
        self.assertFalse(is_denied_mcp_tool("graphiti_search_memory_facts"))


if __name__ == "__main__":
    unittest.main()
