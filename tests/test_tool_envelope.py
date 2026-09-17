"""工具信封归一化单测。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from tools.tool_envelope import is_envelope, normalize_tool_content


class ToolEnvelopeTest(unittest.TestCase):
    def test_kg_dict_wraps(self) -> None:
        raw = json.dumps({"handler": "kg_disease_plain", "items": ["偏头痛"], "summary": "1条"}, ensure_ascii=False)
        out = json.loads(normalize_tool_content("query_consult_graph", raw))
        self.assertTrue(is_envelope(out))
        self.assertTrue(out["usable"])
        self.assertEqual(out["data"]["items"], ["偏头痛"])

    def test_empty_memory_retryable(self) -> None:
        out = json.loads(normalize_tool_content("search_long_memory", "（暂无仍有效的长期事实）"))
        self.assertFalse(out["usable"])
        self.assertTrue(out["retryable"])

    def test_already_envelope(self) -> None:
        env = {
            "ok": True,
            "tool": "x",
            "usable": True,
            "retryable": False,
            "error": "",
            "summary": "s",
            "data": {"items": ["a"]},
        }
        out = json.loads(normalize_tool_content("x", json.dumps(env, ensure_ascii=False)))
        self.assertEqual(out["data"]["items"], ["a"])


if __name__ == "__main__":
    unittest.main()
