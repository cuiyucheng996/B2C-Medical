"""search_exam_handbook 给 Agent 的信封裁剪。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.kg_query import slim_handbook_pack_for_agent
from tools.companion_tools import _kg_dump


class HandbookEnvelopeSlimTest(unittest.TestCase):
    def test_slim_strips_rag_and_caps_items(self) -> None:
        huge = "x" * 5000
        pack = {
            "handler": "kg_symptom_care",
            "source": "ragflow",
            "seeds": ["头痛"],
            "summary": "关于头痛的说明",
            "alignments": [{"mention": "头", "standard_name": "头部"}],
            "rag": {"queries": ["q1", "q2"], "preview": huge},
            "items": [huge, huge, "短条"],
        }
        slim = slim_handbook_pack_for_agent(pack)
        self.assertEqual(slim["summary"], "关于头痛的说明")
        self.assertEqual(len(slim["items"]), 2)
        self.assertLessEqual(len(slim["items"][0]), 300)
        self.assertNotIn("rag", slim)
        self.assertNotIn("alignments", slim)
        self.assertNotIn("seeds", slim)

    def test_kg_dump_envelope_small(self) -> None:
        pack = {
            "handler": "kg_exam_meaning",
            "source": "ragflow",
            "seeds": ["ALT"],
            "summary": "转氨酶",
            "alignments": [],
            "items": ["a" * 4000],
            "rag": {"detail": "y" * 3000},
        }
        text = _kg_dump("search_exam_handbook", pack)
        self.assertLess(len(text), 2500)
        body = json.loads(text)
        data = body["data"]
        self.assertEqual(len(data.get("items") or []), 1)
        self.assertNotIn("preview", json.dumps(data))


if __name__ == "__main__":
    unittest.main()
