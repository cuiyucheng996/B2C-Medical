from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.reply_actions import reply_actions


class ReplyActionsTest(unittest.TestCase):
    def test_hitl_slot_has_no_actions(self) -> None:
        actions = reply_actions(
            hitl={"agent_name": "query_consult_graph", "question": "哪个部位？"},
            preprocess={"intent_major": "公共知识问答"},
        )
        self.assertEqual(actions, [])

    def test_waiting_upload_only_upload(self) -> None:
        actions = reply_actions(
            hitl={"agent_name": "explain_uploaded_report"},
            preprocess={"intent_major": "报告指标解读", "intent_handler": "report_upload"},
        )
        self.assertEqual([item["id"] for item in actions], ["upload_report"])

    def test_kg_with_slot_offers_book(self) -> None:
        actions = reply_actions(
            preprocess={
                "intent_major": "公共知识问答",
                "mentions": {"部位": ["胃"]},
            }
        )
        self.assertEqual([item["id"] for item in actions], ["book_expert"])

    def test_report_done_offers_upload_and_book(self) -> None:
        actions = reply_actions(
            preprocess={"intent_major": "报告指标解读", "intent_handler": "report_explain"},
            last_report={"filename": "a.pdf"},
        )
        self.assertEqual([item["id"] for item in actions], ["upload_report", "book_expert"])

    def test_chat_no_actions(self) -> None:
        actions = reply_actions(preprocess={"intent_major": "闲聊"})
        self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
