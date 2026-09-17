from __future__ import annotations

import sys
import unittest
from io import BytesIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_B2C_SRC = Path(__file__).resolve().parents[1] / "src"
for item in (str(_ROOT), str(_B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)


class ReportImageIngestTest(unittest.TestCase):
    def test_png_stays_raster_for_yolox(self) -> None:
        from PIL import Image

        from algorithm.report_ingest import _ensure_raster_image

        buf = BytesIO()
        Image.new("RGB", (80, 40), "white").save(buf, format="PNG")
        tmp = Path(__file__).resolve().parent / "_tmp_lab.png"
        tmp.write_bytes(buf.getvalue())
        try:
            out = _ensure_raster_image(tmp, buf.getvalue())
            self.assertEqual(out, tmp)
            self.assertFalse(out.read_bytes().startswith(b"%PDF"))
        finally:
            tmp.unlink(missing_ok=True)

    def test_reject_other_suffix(self) -> None:
        from algorithm.file_gate import classify_file_family

        with self.assertRaises(ValueError):
            classify_file_family("a.exe", b"MZ\x90\x00")


if __name__ == "__main__":
    unittest.main()
