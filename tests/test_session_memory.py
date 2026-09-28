"""session_id / user_id 记忆作用域：关场、新开、槽位校验、并行 worker thread_id。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("COMPANION_CHECKPOINTER", "memory")

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))


class SessionMemoryScopeTest(unittest.TestCase):
    def test_memory_scope_dict(self) -> None:
        from algorithm.session_memory import memory_scope_dict

        scope = memory_scope_dict(
            type(
                "S",
                (),
                {"user_id": "u1", "session_id": "s1", "group_id": "u1"},
            )()
        )
        self.assertEqual(scope["user_id"], "u1")
        self.assertEqual(scope["session_id"], "s1")
        self.assertEqual(scope["short_redis"], "conversation:short:s1")
        self.assertEqual(scope["compress_redis"], "conversation:compress:s1")
        self.assertEqual(scope["slot_redis"], "conversation:slots:u1")
        self.assertEqual(scope["graph_thread_id"], "s1")

    def test_companion_graph_thread_id_prefers_session_id(self) -> None:
        from graph.checkpointer import companion_graph_thread_id

        tid = companion_graph_thread_id({"session_id": "sid-99", "user_id": "u1"})
        self.assertEqual(tid, "sid-99")
        fallback = companion_graph_thread_id({"user_id": "u1"})
        self.assertEqual(fallback, "user:u1")

    def test_release_calls_short_compress_checkpoint(self) -> None:
        from algorithm.session_memory import release_session_runtime_state

        with (
            mock.patch("algorithm.short_memory.clear_b2c_short_memory") as clear_redis,
            mock.patch("graph.checkpoint_lifecycle.release_on_session_end") as clear_cp,
        ):
            release_session_runtime_state("u1", "s1", clear_slots=True, clear_checkpoint=True)
        clear_redis.assert_called_once_with("u1", session_id="s1", clear_slots=True)
        clear_cp.assert_called_once_with("s1")

    def test_create_new_releases_old_session_not_slots(self) -> None:
        from session import CompanionSession, SessionStore

        old = CompanionSession(session_id="old-sid", user_id="u1", created_at="t")
        store = SessionStore()
        store._by_user["u1"] = old
        with mock.patch("algorithm.session_memory.release_session_runtime_state") as release:
            new = store.create_new("u1")
        release.assert_called_once_with("u1", "old-sid", clear_slots=False, clear_checkpoint=True)
        self.assertNotEqual(new.session_id, "old-sid")
        self.assertEqual(store.get_active_for_user("u1"), new)

    def test_release_companion_session_always_clears_runtime(self) -> None:
        from algorithm.session_release import release_companion_session
        from session import CompanionSession, SessionStore

        session = CompanionSession(session_id="s-close", user_id="u1", created_at="t")
        store = SessionStore()
        store._by_user["u1"] = session
        with (
            mock.patch(
                "memory.graphiti_store.persist_session_findings",
                return_value={"ok": True, "skipped": "already_persisted", "ingested": 0},
            ),
            mock.patch("algorithm.user_ui_draft.merge_user_ui_draft"),
            mock.patch("algorithm.session_memory.release_session_runtime_state") as release,
        ):
            out = release_companion_session(session, store, reason="idle")
        release.assert_called_once_with("u1", "s-close", clear_slots=True, clear_checkpoint=True)
        self.assertIsNone(store.get_active_for_user("u1"))
        self.assertIn("ingest", out)

    def test_validate_slot_memory_restores_matching_session(self) -> None:
        from algorithm.session_memory import validate_slot_memory_for_session

        empty = {"疾病": [], "症状": [], "部位": []}
        loaded = {"疾病": ["感冒"], "症状": [], "部位": []}
        with mock.patch("algorithm.short_memory.load_b2c_slot_memory", return_value=loaded):
            out = validate_slot_memory_for_session(empty, user_id="u1", session_id="s1")
        self.assertEqual(out["疾病"], ["感冒"])

    def test_slot_load_rejects_stale_session(self) -> None:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.redis.slot_memory import load_slot_memory_sync

        payload = '{"session_id":"other","slots":{"疾病":["a"],"症状":[],"部位":[]}}'
        with mock.patch("src.redis.slot_memory._sync_redis") as redis_factory:
            client = mock.MagicMock()
            client.get.return_value = payload
            redis_factory.return_value = client
            out = load_slot_memory_sync("u1", session_id="current-sid")
        self.assertEqual(out["疾病"], [])

    def test_agent_thread_id_per_parallel_worker(self) -> None:
        from agent.companion_agent import _agent_thread_id

        turn = {"session_id": "s1"}
        a = _agent_thread_id(turn, {"major": "公共知识问答", "handler": "kg_part_diseases"})
        b = _agent_thread_id(turn, {"major": "个人记忆问答", "handler": "mem_companion_fact"})
        self.assertNotEqual(a, b)
        self.assertEqual(a, "agent:s1:公共知识问答:kg_part_diseases")
        self.assertEqual(b, "agent:s1:个人记忆问答:mem_companion_fact")

    def test_clear_b2c_uses_session_id_keys(self) -> None:
        from algorithm.short_memory import clear_b2c_short_memory

        with (
            mock.patch("src.redis.conversation.clear_conversation_sync") as clear_conv,
            mock.patch("src.redis.context_compress.clear_compress_sync") as clear_cmp,
            mock.patch("src.redis.slot_memory.clear_slot_memory_sync") as clear_slots,
            mock.patch("configs.paths_b2c.ensure_uvgraph_root"),
        ):
            clear_b2c_short_memory("u1", session_id="sess-x", clear_slots=False)
        clear_conv.assert_called_once_with("sess-x")
        clear_cmp.assert_called_once_with("sess-x")
        clear_slots.assert_not_called()


if __name__ == "__main__":
    unittest.main()
