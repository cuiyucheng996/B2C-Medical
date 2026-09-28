"""HITL checkpoint 槽位挂起/续跑。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.hitl_checkpoint import (
    consume_slot_hitl,
    get_hitl_checkpointer,
    put_slot_pending,
    slot_thread_id,
)


class HitlCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        get_hitl_checkpointer.cache_clear()
        from algorithm import hitl_checkpoint as hc

        hc._slot_graph.cache_clear()
        hc._meta_graph.cache_clear()

    def test_slot_resume_merges_query(self) -> None:
        sid = "sess-test-1"
        row = put_slot_pending(
            sid,
            question="哪天复查？",
            agent_name="note_task_remind",
            state={"query": "帮我设提醒"},
        )
        tid = slot_thread_id(sid)
        self.assertEqual(row["resume_token"], tid)
        pending = consume_slot_hitl(sid, "", "")
        self.assertIsNotNone(pending)
        done = consume_slot_hitl(sid, tid, "下周三")
        self.assertIsNotNone(done)
        self.assertIn("下周三", str((done or {}).get("state", {}).get("merged_query")))


if __name__ == "__main__":
    unittest.main()
