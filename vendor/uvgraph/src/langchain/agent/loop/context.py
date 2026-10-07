"""对话上下文（Loop 注入用，不绑死 Redis/LangGraph）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.schemas.agent_protocol import Artifact

from .compress import compress_messages, messages_need_compression

__all__ = ["LoopContext"]


@dataclass
class LoopContext:
    system_prompt: str = ""
    messages: list[BaseMessage] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(
        self,
        text: str,
        *,
        tool_calls: list[dict] | None = None,
        think: str = "",
    ) -> None:
        kwargs: dict = {"content": text or ""}
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        extra: dict = {}
        if think:
            extra["reasoning_content"] = think
        if extra:
            kwargs["additional_kwargs"] = extra
        self.messages.append(AIMessage(**kwargs))

    def add_tool_message(self, *, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))

    def add_tool_result(
        self,
        *,
        tool_call_id: str,
        content: str,
        name: str = "tool",
        artifacts: list[Artifact] | None = None,
    ) -> None:
        self.add_tool_message(tool_call_id=tool_call_id, name=name, content=content)
        for art in artifacts or []:
            self.add_artifact(art)

    def add_artifact(self, art: Artifact) -> None:
        self.artifacts.append(art)

    def heal_dangling_tool_calls(self, reason: str) -> int:
        """修复上下文末尾悬空的 tool_calls。

        当 assistant 消息携带 tool_calls 但缺少对应 tool result（如异常
        中断了循环），逐个补写错误结果，保持 assistant(tool_calls) ↔
        tool(result) 配对完整，否则下一轮恢复历史后 DeepSeek 会 400。
        """
        history = self.messages
        last_tc_msg: AIMessage | None = None
        last_tc_idx = -1
        # 从尾部向前找最近一条带 tool_calls 的 assistant 消息；
        # 若先遇到 user/普通 assistant 消息，说明不存在悬空配对。
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if isinstance(msg, ToolMessage):
                continue
            if isinstance(msg, AIMessage) and (getattr(msg, "tool_calls", None) or []):
                last_tc_msg = msg
                last_tc_idx = i
                break
            return 0
        if last_tc_msg is None:
            return 0

        # 统计已有结果的 tool_call_id，仅补缺失项
        answered = {
            str(msg.tool_call_id)
            for msg in history[last_tc_idx + 1 :]
            if isinstance(msg, ToolMessage) and msg.tool_call_id
        }
        content = f"[工具执行中断] {reason}"
        healed = 0
        for call in last_tc_msg.tool_calls or []:
            if isinstance(call, dict):
                call_id = str(call.get("id") or "")
                name = str(call.get("name") or "tool")
            else:
                call_id = str(getattr(call, "id", "") or "")
                name = str(getattr(call, "name", "") or "tool")
            if call_id and call_id not in answered:
                self.add_tool_message(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                )
                healed += 1
        return healed

    def get_messages(self) -> list[BaseMessage]:
        prompt = (self.system_prompt or "").strip()
        if not prompt:
            return list(self.messages)
        return [SystemMessage(content=prompt), *self.messages]

    def needs_compression(self) -> bool:
        return messages_need_compression(self.messages)

    def compress(self) -> None:
        self.messages = compress_messages(self.messages)
