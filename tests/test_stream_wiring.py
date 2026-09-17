"""流式接线单测：单意图才逐 token，多意图并行不串字。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

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


class IntentTaskStreamWiringTest(unittest.IsolatedAsyncioTestCase):
    async def _run(self, tasks: list[dict], on_token) -> list:
        """跑 _ainvoke_intent_tasks，返回每次 _ainvoke_companion_agent 收到的 on_token。"""
        seen: list = []

        async def fake_invoke(model, tools, history, observe, turn=None, task=None, on_token=None):
            seen.append(on_token)
            return f"reply-{task.get('handler')}", []

        with mock.patch.object(companion_agent, "_ainvoke_companion_agent", fake_invoke):
            with mock.patch.object(companion_agent, "mcp_tools_for_major", _no_mcp):
                reply, _facts = await companion_agent._ainvoke_intent_tasks(
                    mock.MagicMock(), tasks, [], {}, on_token=on_token
                )
        return seen, reply

    async def test_single_task_gets_callback(self) -> None:
        tokens: list[str] = []
        seen, reply = await self._run([_task("chat_hello")], tokens.append)
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        seen[0]("片段")  # 回调确实写进了同一个 sink
        self.assertEqual(tokens, ["片段"])
        self.assertEqual(reply, "reply-chat_hello")

    async def test_multi_task_gets_no_callback(self) -> None:
        """并行时不能把 on_token 传下去，否则几个模型会往同一个流里串字。"""
        tokens: list[str] = []
        seen, _reply = await self._run(
            [_task("chat_hello"), _task("chat_thanks")], tokens.append
        )
        self.assertEqual(len(seen), 2)
        for got in seen:
            self.assertIsNone(got)

    async def test_no_token_callback_still_works(self) -> None:
        seen, reply = await self._run([_task("chat_hello")], None)
        self.assertEqual(len(seen), 1)
        self.assertIsNone(seen[0])
        self.assertEqual(reply, "reply-chat_hello")


async def _no_mcp(major: str) -> list:
    return []


if __name__ == "__main__":
    unittest.main()
