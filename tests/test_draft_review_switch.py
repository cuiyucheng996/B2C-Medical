"""子图草稿评审开关。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from graph.review import companion_draft_review_mode, companion_should_review_draft


class DraftReviewSwitchTest(unittest.TestCase):
    def test_default_off(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_DRAFT_REVIEW": ""}, clear=False):
            with mock.patch("graph.review._draft_review_mode_from_yaml", return_value=""):
                self.assertEqual(companion_draft_review_mode(), "off")
        self.assertFalse(companion_should_review_draft("公共知识问答", "kg_symptom_care"))

    def test_public_only(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_DRAFT_REVIEW": "public"}, clear=False):
            self.assertTrue(companion_should_review_draft("公共知识问答", "kg_symptom_care"))
            self.assertFalse(companion_should_review_draft("近况陈述", "status_complaint"))

    def test_all(self) -> None:
        with mock.patch.dict(os.environ, {"COMPANION_DRAFT_REVIEW": "all"}, clear=False):
            self.assertTrue(companion_should_review_draft("近况陈述", "status_complaint"))


if __name__ == "__main__":
    unittest.main()
