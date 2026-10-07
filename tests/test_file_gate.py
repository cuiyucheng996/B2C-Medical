"""文件前置：形态（图/PDF）+ 内容（化验单/普通图/普通文档）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.file_gate import classify_file_family, classify_file_role, intent_for_file_role
from tools.companion_tools import tools_for_intent
from tools.registry import reset_companion_registry


class FileFamilyTest(unittest.TestCase):
    def test_png_magic(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        self.assertEqual(classify_file_family("a.png", data), "image")

    def test_pdf_magic(self) -> None:
        self.assertEqual(classify_file_family("a.pdf", b"%PDF-1.4\n"), "document")

    def test_jpeg_magic(self) -> None:
        self.assertEqual(classify_file_family("a.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 8), "image")

    def test_webp_magic(self) -> None:
        self.assertEqual(classify_file_family("a.webp", b"RIFF\x00\x00\x00\x00WEBP"), "image")

    def test_heic_ftyp(self) -> None:
        data = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8
        self.assertEqual(classify_file_family("IMG_0001.HEIC", data), "image")

    def test_accept_docx(self) -> None:
        self.assertEqual(classify_file_family("a.docx", b"PK\x03\x04xxxx"), "document")

    def test_accept_xlsx(self) -> None:
        self.assertEqual(classify_file_family("a.xlsx", b"PK\x03\x04xxxx"), "document")

    def test_accept_txt(self) -> None:
        self.assertEqual(classify_file_family("note.txt", b"hello"), "document")

    def test_accept_html(self) -> None:
        self.assertEqual(classify_file_family("page.html", b"<html></html>"), "document")

    def test_reject_exe(self) -> None:
        with self.assertRaises(ValueError):
            classify_file_family("a.exe", b"MZ\x90\x00")

    def test_pdf_suffix_but_png_bytes(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        with self.assertRaises(ValueError):
            classify_file_family("fake.pdf", data)


class FileRoleTest(unittest.TestCase):
    def test_table_goes_lab(self) -> None:
        role, reasons = classify_file_role(family="image", table_count=1, indicator_count=0, text="")
        self.assertEqual(role, "lab_report")
        self.assertTrue(reasons)

    def test_indicators_go_lab(self) -> None:
        role, _ = classify_file_role(family="document", table_count=0, indicator_count=5, text="")
        self.assertEqual(role, "lab_report")

    def test_plain_photo(self) -> None:
        role, _ = classify_file_role(family="image", table_count=0, indicator_count=0, text="一只猫坐在沙发上")
        self.assertEqual(role, "general_image")

    def test_pending_scan_with_lab_name(self) -> None:
        role, _ = classify_file_role(
            family="image",
            table_count=0,
            indicator_count=0,
            text="01_血常规_preview.png",
            layout_pending=True,
        )
        self.assertEqual(role, "lab_report")

    def test_pending_photo_without_lab_name(self) -> None:
        role, _ = classify_file_role(
            family="image",
            table_count=0,
            indicator_count=0,
            text="cat.png",
            layout_pending=True,
        )
        self.assertEqual(role, "general_image")

    def test_plain_pdf(self) -> None:
        role, _ = classify_file_role(family="document", table_count=0, indicator_count=1, text="会议纪要")
        self.assertEqual(role, "general_document")

    def test_lab_keyword_alone_not_enough(self) -> None:
        # 只有词 +1 分，不到 2
        role, _ = classify_file_role(family="image", table_count=0, indicator_count=0, text="检验")
        self.assertEqual(role, "general_image")

    def test_keyword_plus_one_table(self) -> None:
        role, _ = classify_file_role(family="document", table_count=1, indicator_count=0, text="参考范围")
        self.assertEqual(role, "lab_report")


class MineruPdfConvertTest(unittest.TestCase):
    def test_no_table_skips_pdf(self) -> None:
        from algorithm.file_gate import _pdf_for_mineru_if_table

        dest = Path("a.png")
        self.assertEqual(_pdf_for_mineru_if_table(dest=dest, probe=dest, table_count=0), "")

    def test_lab_role_png_writes_pdf_without_table(self) -> None:
        import tempfile

        from PIL import Image

        from algorithm.file_gate import _ensure_report_pdf

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "lab.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(dest)
            out = _ensure_report_pdf(dest=dest, probe=dest)
            self.assertTrue(out.endswith(".pdf"))
            self.assertTrue(Path(out).is_file())

    def test_png_with_table_writes_pdf(self) -> None:
        import tempfile

        from PIL import Image

        from algorithm.file_gate import _pdf_for_mineru_if_table

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "shot.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(dest)
            out = _pdf_for_mineru_if_table(dest=dest, probe=dest, table_count=1)
            self.assertTrue(out.endswith(".pdf"))
            self.assertTrue(Path(out).is_file())


class FileEvalTest(unittest.TestCase):
    def test_xlsx_eval_shows_score_and_chunks(self) -> None:
        from io import BytesIO
        from unittest import mock

        from openpyxl import Workbook

        from algorithm.file_eval import run_file_eval

        book = Workbook()
        sheet = book.active
        sheet.append(["项目", "结果", "单位"])
        sheet.append(["总胆固醇", "6.1", "mmol/L"])
        buf = BytesIO()
        book.save(buf)
        with mock.patch(
            "algorithm.file_eval.explain_uploaded_report",
            return_value={"summary": "mocked", "report": {}},
        ):
            row = run_file_eval(filename="lab.xlsx", data=buf.getvalue())
        self.assertEqual(row["role"], "lab_report")
        self.assertEqual(row["intent"]["major"], "报告指标解读")
        self.assertGreaterEqual(int(row["score_card"]["score"]), 2)
        self.assertEqual(row["preprocess"]["artifact"], "structured_chunks")
        self.assertEqual(row["report_split"]["summary"], "mocked")


class FileIngestOfficeTest(unittest.TestCase):
    def test_xlsx_table_routes_lab(self) -> None:
        from io import BytesIO

        from openpyxl import Workbook

        from algorithm.file_gate import ingest_user_file

        book = Workbook()
        sheet = book.active
        sheet.append(["项目", "结果", "单位"])
        sheet.append(["总胆固醇", "6.1", "mmol/L"])
        buf = BytesIO()
        book.save(buf)
        bundle = ingest_user_file(user_id="file-gate-test", filename="lab.xlsx", data=buf.getvalue())
        self.assertEqual(bundle.family, "document")
        self.assertEqual(bundle.parse_source, "preview")
        self.assertEqual(bundle.role, "lab_report")
        self.assertEqual(bundle.table_count, 1)


class DocumentLoadTest(unittest.TestCase):
    def test_txt_loader_stamps_user(self) -> None:
        import tempfile

        from algorithm.document_load import load_user_source
        from algorithm.file_gate import parse_user_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_text("空腹血糖 6.1 mmol/L\n会议纪要草稿", encoding="utf-8")
            loaded = load_user_source(path, user_id="u-doc")
            self.assertEqual(loaded.loader, "txt")
            self.assertIn("会议纪要", loaded.body)
            self.assertEqual(loaded.documents[0].metadata.get("user_id"), "u-doc")
            parsed = parse_user_file(saved_path=str(path), filename="note.txt", family="document", user_id="u-doc")
            self.assertEqual(parsed.parse_source, "txt")
            self.assertEqual(parsed.family, "document")
            self.assertIn("会议纪要", parsed.user_text)
            self.assertTrue(parsed.chunks)
            self.assertEqual(parsed.chunks[0]["user_id"], "u-doc")
            self.assertEqual(parsed.chunks[0]["saved_path"], str(path))
            self.assertIn("category", parsed.chunks[0])
            self.assertIn("page", parsed.chunks[0])

    def test_stamp_layout_elements(self) -> None:
        from algorithm.document_load import stamp_chunks

        chunks = stamp_chunks(
            [{"category": "Table", "page": 2, "text": "血糖 6.1", "html": "<table></table>"}],
            user_id="u-img",
            saved_path="/tmp/a.png",
        )
        self.assertEqual(chunks[0]["user_id"], "u-img")
        self.assertEqual(chunks[0]["saved_path"], "/tmp/a.png")
        self.assertEqual(chunks[0]["page"], 2)
        self.assertEqual(chunks[0]["category"], "Table")


class FileRouteIntentTest(unittest.TestCase):
    def test_mapping(self) -> None:
        self.assertEqual(intent_for_file_role("lab_report")["handler"], "report_upload")
        self.assertEqual(intent_for_file_role("general_image")["handler"], "file_image")
        self.assertEqual(intent_for_file_role("general_document")["handler"], "file_document")
        self.assertEqual(intent_for_file_role("layout_picture")["handler"], "chat_hello")
        self.assertEqual(intent_for_file_role("layout_picture")["major"], "闲聊")


class MineruCacheTest(unittest.TestCase):
    def test_mem_sha_hit_skips_split(self) -> None:
        from unittest.mock import patch

        from algorithm.mineru_mem_cache import file_content_sha256, put_mineru_mem
        from algorithm.report_ingest import mineru_split_tables_cached

        with self._temp_pdf() as path:
            data = path.read_bytes()
            sha = file_content_sha256(data)
            put_mineru_mem(
                sha,
                structured_tables=[{"rows": [["A", "1"]]}],
                indicators=[{"name": "A", "value": "1", "unit": "", "flag": ""}],
                parse_source="mineru_table",
            )
            row = {"content_sha256": sha, "mineru_pdf_path": str(path)}
            with patch("algorithm.report_ingest.mineru_split_tables") as split:
                tables, indicators, source = mineru_split_tables_cached(path, row)
                split.assert_not_called()
            self.assertEqual(len(tables), 1)
            self.assertEqual(indicators[0]["name"], "A")
            self.assertEqual(source, "mineru_table")

    def test_cache_miss_calls_split(self) -> None:
        from unittest.mock import patch

        from algorithm.report_ingest import mineru_split_tables_cached

        with self._temp_pdf() as path:
            with patch(
                "algorithm.report_ingest.mineru_split_tables",
                return_value=([{"rows": []}], [{"name": "B", "value": "2", "unit": "", "flag": ""}], "mineru_table"),
            ) as split:
                tables, indicators, source = mineru_split_tables_cached(path, {})
                split.assert_called_once()
            self.assertEqual(indicators[0]["name"], "B")

    def _temp_pdf(self):
        import tempfile

        class _Ctx:
            def __enter__(self):
                fd, name = tempfile.mkstemp(suffix=".pdf")
                import os

                os.write(fd, b"%PDF-1.4\n1 0 obj\nendobj\n")
                os.close(fd)
                self.path = Path(name)
                return self.path

            def __exit__(self, *args):
                try:
                    self.path.unlink()
                except OSError:
                    pass

        return _Ctx()


class FileToolBindTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_companion_registry()

    def test_image_tool(self) -> None:
        names = [t.name for t in tools_for_intent(major="近况陈述", handler="file_image")]
        self.assertEqual(names, ["describe_uploaded_image"])

    def test_doc_tool(self) -> None:
        names = [t.name for t in tools_for_intent(major="近况陈述", handler="file_document")]
        self.assertEqual(names, ["describe_uploaded_document"])

    def test_layout_picture_chat_no_tools(self) -> None:
        names = [t.name for t in tools_for_intent(major="闲聊", handler="chat_hello")]
        self.assertEqual(names, [])


if __name__ == "__main__":
    unittest.main()
