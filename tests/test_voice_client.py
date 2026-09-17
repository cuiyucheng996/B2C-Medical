from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from algorithm.voice_client import _extract_text


class VoiceClientTest(unittest.TestCase):
    def test_extract_openai_text(self) -> None:
        self.assertEqual(_extract_text({"text": " 头疼 "}), "头疼")

    def test_extract_nested(self) -> None:
        self.assertEqual(_extract_text({"data": {"text": "咳嗽"}}), "咳嗽")


if __name__ == "__main__":
    unittest.main()
