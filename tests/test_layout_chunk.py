"""版面字号认标题切块（不调真实 PDF）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_B2C_SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(_ROOT), str(_B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from parser.layout_chunk import lines_to_chunks


class LayoutChunkTest(unittest.TestCase):
    def test_splits_on_larger_font_not_what_is_regex(self) -> None:
        lines = [
            {"text": "鼻内镜", "size": 16, "bold": True},
            {"text": "用于观察鼻腔与鼻窦黏膜。", "size": 11, "bold": False},
            {"text": "完成鼻内镜检查前需禁食。", "size": 11, "bold": False},
            {"text": "注意事项", "size": 16, "bold": True},
            {"text": "检查后避免擤鼻。", "size": 11, "bold": False},
        ]
        chunks = lines_to_chunks(lines)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("鼻内镜"))
        self.assertIn("观察鼻腔", chunks[0])
        self.assertIn("禁食", chunks[0])
        self.assertTrue(chunks[1].startswith("注意事项"))
        self.assertIn("擤鼻", chunks[1])

    def test_ignores_page_footer(self) -> None:
        lines = [
            {"text": "前言", "size": 15, "bold": True},
            {"text": "这是正文。", "size": 11, "bold": False},
            {"text": "12", "size": 8, "bold": False},
        ]
        chunks = lines_to_chunks(lines)
        self.assertEqual(len(chunks), 1)
        self.assertNotIn("\n12", chunks[0])


if __name__ == "__main__":
    unittest.main()
