from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.coref_rewrite import rewrite_coref_query


class CorefRewriteTest(unittest.TestCase):
    def test_disease_cue_still_works(self) -> None:
        out = rewrite_coref_query(
            "这个病严重吗",
            slot_memory={"疾病": ["胃炎"]},
        )
        self.assertTrue(out.changed)
        self.assertIn("胃炎", out.text)

    def test_metric_cue(self) -> None:
        out = rewrite_coref_query(
            "这个指标偏高危险吗",
            slot_memory={"指标": ["ALT"]},
        )
        self.assertTrue(out.changed)
        self.assertIn("ALT", out.text)
        self.assertNotIn("这个指标", out.text)

    def test_followup_that_danger(self) -> None:
        out = rewrite_coref_query(
            "那危险吗",
            last_user_text="ALT 偏高是什么意思",
            prior_user_texts=["ALT 偏高是什么意思"],
        )
        self.assertTrue(out.changed)
        self.assertIn("ALT", out.text)

    def test_followup_five_items(self) -> None:
        from unittest import mock

        with mock.patch(
            "algorithm.coref_rewrite._llm_complete",
            return_value="不知道孙子所说的五事会怎样",
        ):
            out = rewrite_coref_query(
                "那不知道这五项会怎样？",
                last_user_text="孙子说的「五事」分别是什么？",
                prior_user_texts=["孙子说的「五事」分别是什么？"],
            )
        self.assertTrue(out.changed)
        self.assertEqual(out.source, "llm")
        self.assertIn("五事", out.text)

    def test_new_topic_not_rewritten(self) -> None:
        out = rewrite_coref_query(
            "那我最近失眠怎么办",
            last_user_text="ALT 偏高是什么意思",
            slot_memory={"指标": ["ALT"]},
        )
        self.assertFalse(out.changed)
        self.assertEqual(out.text, "那我最近失眠怎么办")


if __name__ == "__main__":
    unittest.main()
