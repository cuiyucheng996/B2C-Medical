from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.ragflow_eval import run_ragflow_ingest_eval


class RagflowIngestEvalTest(unittest.TestCase):
    def test_chunk_config_override(self) -> None:
        from algorithm.chunk_config import get_chunk_config, set_chunk_config

        before = get_chunk_config()["chunk_token_num"]
        row = set_chunk_config({"chunk_token_num": 128, "local_max_chars": 800})
        self.assertEqual(row["chunk_token_num"], 128)
        self.assertEqual(row["local_max_chars"], 800)
        set_chunk_config({"chunk_token_num": before})
        row = run_ragflow_ingest_eval(filename="a.png", data=b"xx")
        self.assertFalse(row["ok"])
        self.assertEqual(row["steps"][0]["id"], "file")

    def test_happy_path_mocked(self) -> None:
        ping = {"ok": True, "detail": "通"}
        packed = {
            "local": {
                "source": "deepdoc_layout",
                "chunk_count": 2,
                "total_chunks": 2,
                "store": "x",
                "preprocess": {
                    "stages": [
                        {
                            "id": "raster",
                            "title": "转图 + 抽字符",
                            "ok": True,
                            "detail": "1 页",
                            "ops": ["页图 216 dpi", "dedupe_chars 去重"],
                        },
                        {
                            "id": "layout",
                            "title": "版面识别",
                            "ok": True,
                            "detail": "title:1",
                            "ops": ["DeepDoc 出区域", "阈值 0.25"],
                        },
                    ],
                    "samples": [{"type": "title", "page": 1, "text": "鼻内镜"}],
                    "page_rows": [
                        {
                            "page": 1,
                            "chars": 12,
                            "native": 12,
                            "ocr": 0,
                            "backend": "layout",
                            "regions": 1,
                            "lines": 3,
                        }
                    ],
                    "phases": [
                        {
                            "id": "pages",
                            "title": "① 按页转图 + 识别",
                            "aim": "把每页变成「页图 + 字符坐标」两路输入",
                            "branches": [
                                {"cond": "字多（电子版）", "handler": "pdfplumber 抽字符", "note": "按字号聚成行"},
                                {"cond": "字太少（扫描页）", "handler": "OCR 补字", "note": "DeepDoc 优先"},
                            ],
                            "facts": [["页数", "1 页"], ["电子字", "12"]],
                        },
                        {
                            "id": "layout",
                            "title": "② 版面识别（判行 + 判区）",
                            "aim": "先把字符聚成行，再贴类型标签",
                            "branches": [
                                {"cond": "字符 ≥ 120", "handler": "DeepDoc layout.onnx", "note": "逐行出类型"},
                            ],
                            "facts": [["文本行", "3"]],
                        },
                    ],
                },
            },
            "ragflow": {
                "chunk_source": "local",
                "name": "eval__a.pdf",
                "chunk_count": 2,
                "dataset_id": "ds",
                "document_id": "doc",
                "web_url": "http://127.0.0.1:8088",
            },
        }
        with (
            mock.patch("algorithm.ragflow_eval.ping_ragflow", return_value=ping),
            mock.patch("algorithm.ragflow_eval.replace_pdf", return_value=packed),
            mock.patch(
                "algorithm.ragflow_eval.load_local_chunks",
                return_value=[{"doc": "eval__a.pdf", "ord": 0, "page": 1, "content": "鼻内镜检查"}],
            ),
        ):
            row = run_ragflow_ingest_eval(filename="a.pdf", data=b"%PDF-1.4 mock")
        self.assertTrue(row["ok"])
        ids = [s["id"] for s in row["steps"]]
        self.assertEqual(ids[0], "file")
        self.assertIn("raster", ids)
        self.assertIn("layout", ids)
        self.assertIn("ragflow", ids)
        self.assertTrue(all(s["ok"] for s in row["steps"]))
        phase_steps = [s for s in row["steps"] if (s.get("extra") or {}).get("phase")]
        self.assertEqual([s["id"] for s in phase_steps], ["pages", "layout"])
        self.assertTrue(phase_steps[0]["extra"]["branches"])
        self.assertTrue(phase_steps[1]["extra"]["facts"])
        raster = next(s for s in row["steps"] if s["id"] == "raster")
        self.assertTrue(raster["extra"]["ops"])
        self.assertEqual(row["page_rows"][0]["backend"], "layout")
        self.assertEqual(row["phases"][1]["id"], "layout")

    def test_skip_ragflow_runs_local_only(self) -> None:
        """跳过写入时不调 replace_pdf，改为只跑本地分块。"""
        local = {
            "source": "deepdoc_layout",
            "chunk_count": 3,
            "total_chunks": 3,
            "store": "x",
            "preprocess": {"stages": [], "samples": [], "page_rows": [], "phases": []},
        }
        with (
            mock.patch("algorithm.ragflow_eval.ping_ragflow", return_value={"ok": True}),
            mock.patch("algorithm.ragflow_eval.replace_pdf") as replace,
            mock.patch("algorithm.ragflow_eval.ingest_local_pdf", return_value=local) as ingest,
            mock.patch("algorithm.ragflow_eval.load_local_chunks", return_value=[]),
        ):
            row = run_ragflow_ingest_eval(
                filename="a.pdf", data=b"%PDF-1.4 mock", skip_ragflow=True
            )
        replace.assert_not_called()
        ingest.assert_called_once()
        kwargs = ingest.call_args.kwargs
        self.assertEqual(kwargs.get("with_tsr"), False)
        self.assertNotIn("engine", kwargs)
        self.assertTrue(row["ok"])
        self.assertTrue(row["local_only"])
        self.assertTrue(row["skip_ragflow"])
        ragflow_step = next(s for s in row["steps"] if s["id"] == "ragflow")
        self.assertIn("跳过", ragflow_step["detail"])


if __name__ == "__main__":
    unittest.main()
