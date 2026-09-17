"""SpellCheckAgent 编排单测（不调用 LLM）。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from agent.config import SpellCheckConfig
from agent.spell_check_agent import SpellCheckAgent, correct_text
from schema.models import CorrectionResult


class SpellCheckAgentUnitTest(unittest.TestCase):
    def test_empty_text(self) -> None:
        agent = SpellCheckAgent.__new__(SpellCheckAgent)
        agent.config = SpellCheckConfig()
        agent.rouge_scorer = MagicMock()
        result = agent.correct("   ")
        self.assertEqual(result.corrected_text, "")
        self.assertEqual(result.confidence, 1.0)

    def test_low_confidence_rejects_correction(self) -> None:
        runner = SpellCheckAgent.__new__(SpellCheckAgent)
        runner.config = SpellCheckConfig(require_high_confidence=True)
        from agent.confidence import get_rouge_scorer

        runner.rouge_scorer = get_rouge_scorer()

        with patch(
            "agent.spell_check_agent._invoke_spell_agent",
            return_value=CorrectionResult(
                original_text="原文",
                corrected_text="完全改写",
                errors=[],
                confidence=0.95,
                explanation="模型自评很高",
            ),
        ), patch(
            "agent.spell_check_agent.calculate_confidence",
            return_value=0.1,
        ):
            result = runner.correct("原文")

        self.assertEqual(result.corrected_text, "原文")
        self.assertIn("置信度不足", result.explanation)

    def test_correct_text_function_entry(self) -> None:
        with patch("agent.spell_check_agent._get_default_agent") as mock_get:
            mock_runner = MagicMock()
            mock_runner.correct.return_value = CorrectionResult(
                original_text="a",
                corrected_text="a",
                errors=[],
                confidence=1.0,
                explanation="ok",
            )
            mock_get.return_value = mock_runner
            out = correct_text("a")
            self.assertEqual(out.explanation, "ok")


@unittest.skipUnless(
    os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
    "需要配置 LLM API Key",
)
class SpellCheckAgentIntegrationTest(unittest.TestCase):
    def test_no_error_sample(self) -> None:
        corrector = SpellCheckAgent(temperature=0.2)
        result = corrector.correct("今天天气很好。")
        self.assertEqual(result.original_text, "今天天气很好。")


if __name__ == "__main__":
    unittest.main()
