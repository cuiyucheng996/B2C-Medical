"""ROUGE 置信度单测（不依赖 LLM）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from agent.confidence import calculate_confidence, levenshtein_distance


class ConfidenceTest(unittest.TestCase):
    def test_identical_text_full_confidence(self) -> None:
        self.assertEqual(calculate_confidence("abc", "abc"), 1.0)

    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("我喜你", "我喜欢你"), 1)

    def test_minor_edit_has_mid_confidence(self) -> None:
        score = calculate_confidence("我喜你", "我喜欢你")
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.9)


if __name__ == "__main__":
    unittest.main()
