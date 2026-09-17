from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.coref_eval import run_coref_eval, run_coref_one


class CorefEvalSuiteTest(unittest.TestCase):
    def test_gold_all_pass(self) -> None:
        from unittest import mock

        with mock.patch(
            "algorithm.coref_rewrite._llm_complete",
            return_value="不知道孙子所说的五事会怎样",
        ):
            out = run_coref_eval()
        failed = [row["id"] for row in out["rows"] if not row["ok"]]
        self.assertEqual(failed, [], msg=str(out["rows"]))
        self.assertTrue(out["ok"])
        five = next(row for row in out["rows"] if row["id"] == "follow_five")
        self.assertEqual(five["got_source"], "llm")

    def test_one_followup(self) -> None:
        row = run_coref_one(query="那危险吗", prior="ALT 偏高是什么意思")
        self.assertTrue(row["got_changed"])
        self.assertIn("ALT", row["got_text"])


if __name__ == "__main__":
    unittest.main()
