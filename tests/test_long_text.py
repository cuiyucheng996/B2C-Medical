"""长文本分块与合并单测（不依赖 LLM）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from agent.long_text import merge_corrections, split_long_text
from schema.models import CorrectionResult, ErrorDetail


class LongTextTest(unittest.TestCase):
    def test_short_text_no_split(self) -> None:
        text = "今天天气很好。"
        self.assertEqual(split_long_text(text), [text])

    def test_long_text_splits_by_sentence(self) -> None:
        text = "今天天气很好。" * 50
        chunks = split_long_text(text, max_length=300)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 320)

    def test_merge_offsets_error_positions(self) -> None:
        chunk_a = "我喜你。"
        chunk_b = "今天很好。"
        r1 = CorrectionResult(
            original_text=chunk_a,
            corrected_text="我喜欢你。",
            errors=[
                ErrorDetail(
                    position=1,
                    error_span="喜",
                    error_type="拼写错误",
                    correction="喜欢",
                    reason="缺字",
                )
            ],
            confidence=0.9,
            explanation="块1",
        )
        r2 = CorrectionResult(
            original_text=chunk_b,
            corrected_text=chunk_b,
            errors=[],
            confidence=1.0,
            explanation="块2",
        )
        merged = merge_corrections([chunk_a, chunk_b], [r1, r2])
        self.assertEqual(merged.errors[0].position, 1)
        self.assertEqual(len(merged.chunks), 2)
        self.assertIn("我喜欢你", merged.corrected_text)


if __name__ == "__main__":
    unittest.main()
