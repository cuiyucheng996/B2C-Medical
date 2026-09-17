"""陪聊工具装配单测（不调 LLM）。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from tools.companion_tools import (
    companion_turn,
    note_task_appointment,
    note_task_checkin,
    note_task_medication,
    note_task_remind,
    note_task_report,
    tools_for_intent,
)


class CompanionToolsetTest(unittest.TestCase):
    def test_memory_stated_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="个人记忆问答", handler="mem_stated_consult")]
        self.assertEqual(names, ["lookup_stated_consult"])

    def test_memory_life_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="个人记忆问答", handler="mem_companion_fact")]
        self.assertEqual(names, ["lookup_life_facts"])

    def test_kg_binds_graph(self) -> None:
        names = [t.name for t in tools_for_intent(major="公共知识问答", handler="kg_part_diseases")]
        self.assertEqual(names, ["query_consult_graph"])

    def test_kg_binds_handbook(self) -> None:
        names = [t.name for t in tools_for_intent(major="公共知识问答", handler="kg_exam_meaning")]
        self.assertEqual(names, ["search_exam_handbook", "search_milvus_knowledge"])

    def test_kg_care_handbook_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="公共知识问答", handler="kg_symptom_care")]
        self.assertEqual(names, ["search_exam_handbook"])

    def test_task_remind_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="task_remind")]
        self.assertEqual(names, ["note_task_remind"])

    def test_task_checkin_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="task_checkin")]
        self.assertEqual(names, ["note_task_checkin"])

    def test_task_medication_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="task_medication")]
        self.assertEqual(names, ["note_task_medication"])

    def test_task_appointment_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="task_appointment")]
        self.assertEqual(names, ["note_task_appointment"])

    def test_task_report_only(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="task_report")]
        self.assertEqual(names, ["note_task_report"])

    def test_task_default_lists_all_five(self) -> None:
        names = [t.name for t in tools_for_intent(major="具体任务", handler="")]
        self.assertEqual(
            names,
            [
                "note_task_remind",
                "note_task_checkin",
                "note_task_medication",
                "note_task_appointment",
                "note_task_report",
            ],
        )

    def test_task_kinds_cover_n8n_switch(self) -> None:
        """kind 是 n8n Switch 的分支键，两端必须一致。"""
        from tools.companion_tools import TASK_KINDS

        self.assertEqual(
            set(TASK_KINDS),
            {"remind", "checkin", "medication", "appointment", "report"},
        )

    def test_chat_no_tools(self) -> None:
        self.assertEqual(tools_for_intent(major="闲聊", handler="chat_hello"), [])

    def test_registry_lists_mcp(self) -> None:
        from tools.registry import list_companion_registry

        names = {row["name"] for row in list_companion_registry()}
        self.assertIn("query_consult_graph", names)
        self.assertIn("search_exam_handbook", names)
        self.assertIn("search_milvus_knowledge", names)
        self.assertIn("mcp:time", names)
        self.assertIn("mcp:fetch", names)
        self.assertIn("mcp:graphiti", names)

    def test_report_toolkit_is_domain_pack(self) -> None:
        from tools.toolkits import REPORT_TOOLKIT

        self.assertEqual(
            list(REPORT_TOOLKIT.names()),
            [t.name for t in tools_for_intent(major="报告指标解读", handler="")],
        )
        self.assertEqual(
            list(REPORT_TOOLKIT.pick("explain_uploaded_report", "extract_report_tables")),
            [t.name for t in tools_for_intent(major="报告指标解读", handler="report_upload")],
        )


class TaskToolPushTest(unittest.TestCase):
    """5 个具体任务工具 → n8n payload 的槽位映射（n8n 调用被 mock）。"""

    def _push(self, func, args: dict) -> tuple[dict, dict]:
        token = companion_turn.set(
            {"user_id": "u-test", "session_id": "s-1", "query": "用户原话", "handler": "task"}
        )
        try:
            with mock.patch(
                "algorithm.n8n_task.push_companion_task",
                return_value={"ok": True, "status": 200},
            ) as fake:
                envelope = json.loads(func.invoke(args))
        finally:
            companion_turn.reset(token)
        return envelope, fake.call_args.kwargs

    def test_remind_pushes_kind_remind(self) -> None:
        envelope, sent = self._push(note_task_remind, {"when": "下周三"})
        self.assertEqual(sent["kind"], "remind")
        self.assertEqual(sent["when_text"], "下周三")
        self.assertTrue(envelope["usable"])
        self.assertFalse(envelope["retryable"])

    def test_checkin_pushes_kind_checkin(self) -> None:
        _envelope, sent = self._push(note_task_checkin, {"what": "今天药吃完了"})
        self.assertEqual(sent["kind"], "checkin")

    def test_medication_joins_drug_and_time(self) -> None:
        envelope, sent = self._push(
            note_task_medication, {"drug": "二甲双胍", "when": "早饭后"}
        )
        self.assertEqual(sent["kind"], "medication")
        self.assertIn("二甲双胍", sent["title"])
        self.assertIn("早饭后", sent["title"])
        self.assertEqual(envelope["data"]["drug"], "二甲双胍")

    def test_appointment_joins_department_and_time(self) -> None:
        envelope, sent = self._push(
            note_task_appointment, {"department": "内分泌科", "when": "10月8日"}
        )
        self.assertEqual(sent["kind"], "appointment")
        self.assertIn("内分泌科", sent["title"])
        self.assertEqual(envelope["data"]["department"], "内分泌科")

    def test_report_joins_exam_and_time(self) -> None:
        envelope, sent = self._push(
            note_task_report, {"exam": "糖化血红蛋白", "when": "下周一"}
        )
        self.assertEqual(sent["kind"], "report")
        self.assertIn("糖化血红蛋白", sent["title"])
        self.assertEqual(envelope["data"]["exam"], "糖化血红蛋白")

    def test_user_id_propagates(self) -> None:
        _envelope, sent = self._push(note_task_remind, {"when": "下周"})
        self.assertEqual(sent["user_id"], "u-test")
        self.assertEqual(sent["session_id"], "s-1")


if __name__ == "__main__":
    unittest.main()
