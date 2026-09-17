from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.unstructured_tables import describe_unstructured_preprocess, elements_to_chunk_records, extract_tables_unstructured


class UnstructuredTablesTest(unittest.TestCase):
    def test_table_becomes_html_chunk(self) -> None:
        records = elements_to_chunk_records(
            [
                {"category": "Title", "page": 1, "text": "血常规", "html": ""},
                {"category": "Table", "page": 1, "text": "WBC", "html": "<table><tr><td>WBC</td></tr></table>"},
                {"category": "NarrativeText", "page": 1, "text": "备注", "html": ""},
            ]
        )
        self.assertEqual(len(records), 3)
        self.assertIn("<table", records[1]["content"])

    def test_phases_present(self) -> None:
        info = describe_unstructured_preprocess(
            [{"category": "Table", "page": 1, "text": "a", "html": "<table></table>"}],
            path=Path("a.pdf"),
        )
        self.assertEqual([p["id"] for p in info["phases"]], ["pages", "layout"])
        self.assertTrue(info["page_rows"])

    def test_extract_uses_loader(self) -> None:
        dummy = {
            "ok": True,
            "elements": [
                {
                    "category": "Table",
                    "page": 1,
                    "text": "WBC",
                    "html": "<table><tr><td>WBC</td></tr></table>",
                }
            ],
        }
        with mock.patch("algorithm.unstructured_tables._run_layout", return_value=dummy):
            row = extract_tables_unstructured(Path(__file__))
        self.assertTrue(row["ok"])
        self.assertEqual(row["table_count"], 1)
        self.assertTrue(row["preprocess"]["phases"])


if __name__ == "__main__":
    unittest.main()
