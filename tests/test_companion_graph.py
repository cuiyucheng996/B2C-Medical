"""LangGraph 陪聊图能编译；评审打回会再跑执行节点。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ["COMPANION_USE_SUPERVISOR"] = "0"
os.environ["COMPANION_CHECKPOINTER"] = "memory"

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from graph.companion import build_companion_graph, dispatch_intents, merge_node, _merge_worker_outputs_for_trace
from graph.intent_worker import route_after_review
from graph.peer_handoff import format_peer_brief, route_after_intent_worker
from graph.state import CompanionGraphState


class CompanionGraphBuildTest(unittest.TestCase):
    def test_compile(self) -> None:
        graph = build_companion_graph()
        names = set(graph.get_graph().nodes)
        self.assertIn("intent_worker", names)
        self.assertIn("merge", names)
        self.assertIn("peer_handoff", names)
        self.assertIn("begin_turn", names)

    def test_dispatch_fanout(self) -> None:
        state: CompanionGraphState = {
            "query": "hi",
            "observe": "",
            "history": [],
            "base_turn": {},
            "tasks": [
                {"major": "闲聊", "intent": "寒暄打招呼", "handler": "chat_hello", "observe": ""},
                {"major": "公共知识问答", "intent": "部位有哪些病", "handler": "kg_part_diseases", "observe": ""},
            ],
        }
        sends = dispatch_intents(state)
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0].node, "intent_worker")
        self.assertEqual(sends[1].node, "intent_worker")

    def test_peer_brief_excludes_self(self) -> None:
        outputs = [
            {"major": "闲聊", "intent": "a", "handler": "h1", "reply": "你好"},
            {"major": "公共知识问答", "intent": "b", "handler": "h2", "reply": "知识"},
        ]
        brief = format_peer_brief(outputs, exclude=("闲聊", "a", "h1"))
        self.assertIn("知识", brief)
        self.assertNotIn("你好", brief)

    def test_route_peer_handoff_multi_task(self) -> None:
        state: CompanionGraphState = {
            "handoff_round": 0,
            "tasks": [{"major": "闲聊"}, {"major": "公共知识问答"}],
            "worker_outputs": [{"major": "闲聊", "reply": "x"}],
        }
        self.assertEqual(route_after_intent_worker(state), "peer_handoff")
        state["handoff_round"] = 1
        self.assertEqual(route_after_intent_worker(state), "merge")

    def test_review_routes(self) -> None:
        self.assertEqual(route_after_review({"review_pass": True, "review_round": 0}), "finish")
        self.assertEqual(route_after_review({"review_pass": False, "review_round": 1}), "execute")
        self.assertEqual(route_after_review({"review_pass": False, "review_round": 2}), "finish")

    def test_merge_handoff_tool_trace_from_wave1(self) -> None:
        wave1 = [
            {
                "major": "近况陈述",
                "intent": "i1",
                "handler": "h1",
                "bound_tools": ["search_long_memory"],
                "tool_trace": [{"name": "search_long_memory", "args": "{}", "summary": "ok", "ok": "是"}],
                "middleware_trace": [],
                "reply": "初稿",
            }
        ]
        wave2 = [
            {
                "major": "近况陈述",
                "intent": "i1",
                "handler": "h1",
                "bound_tools": ["search_long_memory"],
                "tool_trace": [],
                "middleware_trace": [],
                "reply": "终稿",
                "review_pass": True,
            }
        ]
        merged = _merge_worker_outputs_for_trace(wave1, wave2)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["tool_trace"]), 1)
        self.assertEqual(merged[0]["tool_trace"][0]["name"], "search_long_memory")
        self.assertEqual(merged[0]["reply"], "终稿")

        state: CompanionGraphState = {
            "worker_outputs": wave2,
            "wave1_outputs": wave1,
            "thread_id": "t1",
            "turn_count": 1,
        }
        out = merge_node(state)
        runs = (out.get("trace") or {}).get("tool_runs") or []
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0].get("steps") or []), 1)


class CompanionGraphCheckpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_released_after_run(self) -> None:
        from graph.checkpointer import companion_invoke_config, get_companion_checkpointer
        from graph.companion import build_companion_graph

        get_companion_checkpointer.cache_clear()
        build_companion_graph.cache_clear()
        graph = build_companion_graph()
        tid = "test-thread-release"
        config = companion_invoke_config(tid)
        task = [{"major": "闲聊", "intent": "寒暄打招呼", "handler": "chat_hello", "observe": "观察"}]

        async def fake_invoke(model, tools, history, observe, turn=None, task=None, on_token=None):
            return "你好", []

        with mock.patch("agent.companion_agent._ainvoke_companion_agent", fake_invoke):
            with mock.patch("graph.intent_worker.mcp_tools_for_major", mock.AsyncMock(return_value=[])):
                with mock.patch("graph.intent_worker.tools_for_intent", lambda **_k: []):
                    with mock.patch("graph.intent_worker.review_companion_draft", lambda **_k: {"pass": True, "reason": "", "hint": ""}):
                        from graph.companion import ainvoke_send_companion_graph

                        _reply, _facts, trace = await ainvoke_send_companion_graph(
                            mock.MagicMock(),
                            task,
                            [],
                            {"query": "你好", "session_id": tid},
                            thread_id=tid,
                        )
        self.assertEqual(trace.get("checkpoint_retention"), "run")
        snap = await graph.aget_state(config)
        self.assertEqual(dict(snap.values or {}) if snap else {}, {})


class CompanionGraphRetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_fail_then_pass_runs_execute_twice(self) -> None:
        from graph.companion import ainvoke_companion_intent_graph

        calls: list[str] = []
        reviews = iter(
            [
                {"pass": False, "reason": "跑题", "hint": "对准寒暄"},
                {"pass": True, "reason": "ok", "hint": ""},
            ]
        )

        async def fake_invoke(model, tools, history, observe, turn=None, task=None, on_token=None):
            calls.append(observe)
            return "你好呀", []

        with mock.patch("agent.companion_agent._ainvoke_companion_agent", fake_invoke):
            with mock.patch("graph.intent_worker.mcp_tools_for_major", mock.AsyncMock(return_value=[])):
                with mock.patch("graph.intent_worker.tools_for_intent", lambda **_k: []):
                    with mock.patch("graph.intent_worker.review_companion_draft", lambda **_k: next(reviews)):
                        from graph.companion import ainvoke_send_companion_graph

                        reply, _facts, trace = await ainvoke_send_companion_graph(
                            mock.MagicMock(),
                            [{"major": "闲聊", "intent": "寒暄打招呼", "handler": "chat_hello", "observe": "观察"}],
                            [],
                            {"query": "你好"},
                        )
        self.assertEqual(len(calls), 2)
        self.assertIn("评审打回", calls[1])
        self.assertEqual(reply, "你好呀")
        self.assertEqual(trace.get("workers"), 1)
        self.assertTrue((trace.get("reviews") or [{}])[0].get("pass"))


if __name__ == "__main__":
    unittest.main()
