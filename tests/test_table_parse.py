from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
_B2C_SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(_ROOT), str(_B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from algorithm.report_ingest import (
    _compose_limited_markdown,
    _extract_indicators,
    _html_table_rows,
    _join_whole_tables,
    _pdf_extract_module,
)


class TableParseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _pdf_extract_module()

    def test_html_table_not_pipe_only(self) -> None:
        md = "# 血常规\n\n<table><tr><td>白细胞</td><td>12.1</td></tr></table>\n\na | b\n"
        tables = self.mod.extract_markdown_tables(md)
        self.assertEqual(len(tables), 1)
        self.assertIn("<table>", tables[0])
        self.assertIn("血常规", tables[0])

    def test_code_pipe_is_not_table(self) -> None:
        md = "```python\nrag = (func1 | func2)\n```\n"
        self.assertEqual(self.mod.extract_markdown_tables(md), [])

    def test_rowspan_fills_name(self) -> None:
        html = (
            "<table><tr><td rowspan=2>血常规</td><td>白细胞</td><td>12</td></tr>"
            "<tr><td>红细胞</td><td>4</td></tr></table>"
        )
        rows = _html_table_rows(html)
        self.assertEqual(rows[0][0], "血常规")
        self.assertEqual(rows[1][0], "血常规")

    def test_indicators_keep_table_group(self) -> None:
        table = "## 血常规\n\n<table><tr><td>白细胞</td><td>12.1 10^9/L ↑</td></tr></table>"
        found = _extract_indicators([table], "")
        self.assertTrue(found)
        self.assertEqual(found[0]["name"], "白细胞")
        self.assertEqual(found[0].get("group"), "血常规")

    def test_compose_does_not_cut_table(self) -> None:
        table = "<table>" + ("<tr><td>白细胞</td><td>1</td></tr>" * 20) + "</table>"
        parsed = SimpleNamespace(tables=[table], body="x" * 8000, titles=["化验"], markdown="", image_count=0)
        md = _compose_limited_markdown(parsed)
        self.assertIn("</table>", md)
        self.assertIn("正文已截断", md)

    def test_join_whole_tables(self) -> None:
        blob = _join_whole_tables(["<table>aaaa</table>", "<table>bbbb</table>"], budget=20)
        self.assertEqual(blob, "<table>aaaa</table>")
        self.assertNotIn("bbbb", blob)


if __name__ == "__main__":
    unittest.main()
