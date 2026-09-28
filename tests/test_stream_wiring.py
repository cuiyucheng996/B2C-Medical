"""流式接线：单意图子图 CompanionReply；多意图谁先完谁先推 SSE。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ["COMPANION_USE_SUPERVISOR"] = "0"

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

import agent.companion_agent as companion_agent


def _task(handler: str) -> dict:
    return {
        "major": "闲聊",
        "intent": "寒暄打招呼",
        "handler": handler,
        "tools": [],
        "observe": "",
    }


def _pass_review(**_kwargs: object) -> dict:
    return {"pass": True, "reason": "ok", "hint": ""}


async def _no_mcp(major: str) -> list:
    return []


class IntentTaskStreamWiringTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, tasks: list[dict], on_token) -> tuple[list, str]:
        seen: list = []

        async def fake_invoke(model, tools, history, observe, turn=None, task=None, on_token=None):
            seen.append(on_token)
            return f"reply-{task.get('handler')}", [], []

        with mock.patch.object(companion_agent, "_ainvoke_companion_agent", fake_invoke):
            with mock.patch("graph.intent_worker.mcp_tools_for_major", _no_mcp):
                with mock.patch("graph.intent_worker.tools_for_intent", lambda **_k: []):
                    with mock.patch("graph.intent_worker.review_companion_draft", _pass_review):
                        reply, _facts = await companion_agent._ainvoke_intent_tasks(
                            mock.MagicMock(), tasks, [], {}, on_token=on_token
                        )
        return seen, reply

    async def test_single_task_gets_callback(self) -> None:
        tokens: list[str] = []
        seen, reply = await self._run([_task("chat_hello")], tokens.append)
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        seen[0]("片段")
        self.assertEqual(tokens, ["片段"])
        self.assertEqual(reply, "reply-chat_hello")

    async def test_multi_task_streams_when_worker_done(self) -> None:
        tokens: list[str] = []
        seen, reply = await self._run(
            [_task("chat_hello"), _task("chat_thanks")], tokens.append
        )
        self.assertEqual(len(seen), 2)
        for got in seen:
            self.assertTrue(got is None or callable(got))
        joined = "".join(tokens)
        self.assertIn("reply-chat_hello", joined)
        self.assertIn("reply-chat_thanks", joined)
        self.assertIn("reply-chat_hello", reply)
        self.assertIn("reply-chat_thanks", reply)

    async def test_multi_task_forwards_model_tokens(self) -> None:
        tokens: list[str] = []

        async def fake_invoke(model, tools, history, observe, turn=None, task=None, on_token=None):
            text = f"reply-{task.get('handler')}"
            if on_token:
                for ch in text:
                    on_token(ch)
            return text, [], []

        with mock.patch.object(companion_agent, "_ainvoke_companion_agent", fake_invoke):
            with mock.patch("graph.intent_worker.mcp_tools_for_major", _no_mcp):
                with mock.patch("graph.intent_worker.tools_for_intent", lambda **_k: []):
                    with mock.patch("graph.intent_worker.review_companion_draft", _pass_review):
                        await companion_agent._ainvoke_intent_tasks(
                            mock.MagicMock(),
                            [_task("chat_hello"), _task("chat_thanks")],
                            [],
                            {},
                            on_token=tokens.append,
                        )
        pieces = [item for item in tokens if item != "\n\n"]
        self.assertTrue(all(len(item) == 1 for item in pieces))
        self.assertIn("r", pieces)
        self.assertEqual("".join(pieces).count("reply-"), 2)

    async def test_no_token_callback_still_works(self) -> None:
        seen, reply = await self._run([_task("chat_hello")], None)
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0])
        self.assertEqual(reply, "reply-chat_hello")


if __name__ == "__main__":
    unittest.main()
