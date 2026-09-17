"""报告解读三工具：绑定、缺槽 HITL、无上传件时不可用。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.companion_hitl import ask_if_needed
from algorithm.report_read import compare_lab_history, explain_uploaded_report
from tools.companion_tools import companion_turn, tools_for_intent
from tools.registry import reset_companion_registry


class _Prep:
    def __init__(self, major: str, handler: str, text: str, mentions: dict | None = None) -> None:
        self.intent_major = major
        self.intent_handler = handler
        self.resolved_text = text
        self.corrected_text = text
        self.mentions = mentions or {}
        self.alignments = []


class ReportToolBindTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_companion_registry()

    def test_upload_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="报告指标解读", handler="report_upload")]
        self.assertEqual(names, ["explain_uploaded_report", "extract_report_tables"])

    def test_tables_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="报告指标解读", handler="report_tables")]
        self.assertEqual(names, ["extract_report_tables"])

    def test_explain_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="报告指标解读", handler="report_explain")]
        self.assertEqual(names, ["explain_mentioned_report"])

    def test_compare_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="报告指标解读", handler="report_compare")]
        self.assertEqual(names, ["compare_lab_history"])

    def test_default_lists_three(self) -> None:
        names = [t.name for t in tools_for_intent(major="报告指标解读", handler="")]
        self.assertEqual(
            names,
            [
                "explain_uploaded_report",
                "extract_report_tables",
                "explain_mentioned_report",
                "compare_lab_history",
            ],
        )


class ReportHitlTest(unittest.TestCase):
    def test_upload_without_file_asks(self) -> None:
        asked = ask_if_needed(_Prep("报告指标解读", "report_upload", "帮我看看化验单"))
        self.assertIsNotNone(asked)
        self.assertEqual(asked[1], "explain_uploaded_report")

    def test_upload_with_saved_file_skips(self) -> None:
        self.assertIsNone(
            ask_if_needed(
                _Prep("报告指标解读", "report_upload", "帮我看看化验单"),
                has_uploaded_file=True,
            )
        )

        self.assertIsNone(
            ask_if_needed(
                _Prep("报告指标解读", "report_upload", "【上传检查报告】a.pdf\n尚未抽表"),
            )
        )

    def test_explain_vague_asks(self) -> None:
        asked = ask_if_needed(_Prep("报告指标解读", "report_explain", "那个怎么看"))
        self.assertEqual(asked[1], "explain_mentioned_report")

    def test_explain_with_report_word_skips(self) -> None:
        self.assertIsNone(ask_if_needed(_Prep("报告指标解读", "report_explain", "帮我看看血常规报告")))

    def test_compare_without_metric_asks(self) -> None:
        asked = ask_if_needed(_Prep("报告指标解读", "report_compare", "和上次比比"))
        self.assertEqual(asked[1], "compare_lab_history")

    def test_compare_with_metric_skips(self) -> None:
        self.assertIsNone(ask_if_needed(_Prep("报告指标解读", "report_compare", "胆固醇和上次比")))


class ReportReadTest(unittest.TestCase):
    def test_upload_empty_without_file(self) -> None:
        out = explain_uploaded_report(query="看看", last_report={})
        self.assertTrue(out["empty"])

    def test_upload_uses_preprocess_chunks(self) -> None:
        with mock.patch("algorithm.report_read._handbook", return_value={"items": [], "source": ""}):
            out = explain_uploaded_report(
                query="解读",
                last_file={
                    "filename": "a.png",
                    "saved_path": "/tmp/a.png",
                    "user_id": "u1",
                    "table_count": 1,
                    "tables": ["<table><tr><td>白细胞</td><td>12.4</td></tr></table>"],
                    "chunks": [
                        {
                            "user_id": "u1",
                            "saved_path": "/tmp/a.png",
                            "page": 1,
                            "category": "Table",
                            "text": "白细胞 12.4",
                            "html": "<table><tr><td>白细胞</td><td>12.4</td></tr></table>",
                        }
                    ],
                },
            )
        self.assertFalse(out["empty"])
        self.assertEqual(out["report"]["filename"], "a.png")
        self.assertGreaterEqual(out["report"]["table_count"], 1)
        self.assertTrue(out["report"]["indicators"])
        self.assertEqual(out["report"]["chunks"][0]["page"], 1)
        with mock.patch("algorithm.report_read._handbook", return_value={"items": ["ALT 参考 7-40"], "source": "ragflow"}):
            out = explain_uploaded_report(
                query="解读",
                last_report={"filename": "a.pdf", "indicators": [{"name": "丙氨酸氨基转移酶", "value": "68"}]},
            )
        self.assertFalse(out["empty"])
        self.assertEqual(out["report"]["filename"], "a.pdf")
        self.assertTrue(out["items"])

    def test_compare_need_metric(self) -> None:
        out = compare_lab_history(metric="", query="", user_id="", last_report={})
        self.assertTrue(out["need_metric"])
        self.assertTrue(out["empty"])

    def test_compare_matches_current(self) -> None:
        with mock.patch("algorithm.report_read._facts", return_value=[{"name": "胆固醇", "fact": "上次 5.2"}]):
            with mock.patch("algorithm.report_read._handbook", return_value={"items": [], "source": ""}):
                out = compare_lab_history(
                    metric="胆固醇",
                    query="胆固醇和上次比",
                    user_id="u1",
                    last_report={"indicators": [{"name": "总 胆固醇", "value": "6", "unit": "mmol/L"}]},
                )
        self.assertFalse(out["empty"])
        self.assertEqual(out["current"][0]["value"], "6")


class ReportToolEnvelopeTest(unittest.TestCase):
    def test_upload_tool_without_report(self) -> None:
        from tools.companion_tools import explain_uploaded_report as tool_fn

        tok = companion_turn.set({"query": "看看", "last_report": {}})
        try:
            raw = tool_fn.invoke({"query": "看看"})
        finally:
            companion_turn.reset(tok)
        payload = json.loads(raw)
        self.assertFalse(payload["usable"])
        self.assertEqual(payload["tool"], "explain_uploaded_report")


if __name__ == "__main__":
    unittest.main()
