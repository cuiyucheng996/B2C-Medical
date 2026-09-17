"""编号条目切块与隐形数字层。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_B2C_SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(_ROOT), str(_B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from parser.content_pipeline import content_blocks_to_chunk_records
from parser.line_merge import _drop_overlay_chars
from parser.text_clean import is_catalog_item_line


class CatalogChunkTest(unittest.TestCase):
    def test_item_lines(self) -> None:
        self.assertTrue(is_catalog_item_line("7. 定期进行健康体检。"))
        self.assertTrue(is_catalog_item_line("10. 儿童出生后应按照免疫程序接种疫苗"))
        self.assertTrue(is_catalog_item_line("一、基本知识和理念"))
        self.assertFalse(is_catalog_item_line("定期进行健康体检，了解身体"))

    def test_split_seven_and_eight(self) -> None:
        blocks = [
            {"type": "text", "page": 6, "bbox": [0, 0, 10, 10], "text": "7. 定期进行健康体检。"},
            {"type": "text", "page": 6, "bbox": [0, 20, 10, 40], "text": "定期进行健康体检，了解身体健康状况。"},
            {"type": "text", "page": 6, "bbox": [0, 50, 10, 60], "text": "8. 血压、体温、呼吸和心率是人体的四大生命体征。"},
            {"type": "text", "page": 6, "bbox": [0, 70, 10, 90], "text": "成年人正常血压为收缩压。"},
        ]
        chunks = content_blocks_to_chunk_records(blocks)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0]["content"].startswith("7."))
        self.assertNotIn("8.", chunks[0]["content"])
        self.assertTrue(chunks[1]["content"].startswith("8."))

    def test_drop_overlay_font(self) -> None:
        body = [
            {"text": "十", "fontname": "Song", "top": 80, "x0": 10, "x1": 20, "height": 12}
            for _ in range(50)
        ]
        overlay = [
            {"text": str(i % 10), "fontname": "DejaVuSans", "top": 9.0, "x0": float(i), "x1": float(i) + 1, "height": 8}
            for i in range(40)
        ]
        kept = _drop_overlay_chars(body + overlay)
        self.assertTrue(all(c["fontname"] != "DejaVuSans" for c in kept))
        self.assertEqual(len(kept), 50)


if __name__ == "__main__":
    unittest.main()
