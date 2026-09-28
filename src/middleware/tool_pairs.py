"""发给模型前丢掉对不上的 ToolMessage，避免 OpenAI 兼容口 400。"""



from __future__ import annotations



from typing import Any



from langchain.agents.middleware.types import AgentMiddleware, AgentState

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES



__all__ = [

    "CompanionToolPairMiddleware",

    "clear_agent_checkpoint_thread",

    "drop_orphan_tool_messages",

]





def _call_ids(msg: AIMessage) -> set[str]:

    ids: set[str] = set()

    for item in getattr(msg, "tool_calls", None) or []:

        if isinstance(item, dict):

            tid = str(item.get("id") or "")

        else:

            tid = str(getattr(item, "id", "") or "")

        if tid:

            ids.add(tid)

    return ids





def _ai_without_tool_calls(msg: AIMessage) -> AIMessage:

    content = msg.content

    if isinstance(content, list):

        parts = []

        for block in content:

            if isinstance(block, dict) and block.get("type") == "text":

                parts.append(str(block.get("text") or ""))

            elif isinstance(block, str):

                parts.append(block)

        content = "".join(parts)

    text = str(content or "").strip()

    return AIMessage(content=text or "（继续）")





def _coerce_message(msg: Any) -> BaseMessage | None:

    if isinstance(msg, BaseMessage):

        return msg

    if not isinstance(msg, dict):

        return None

    role = str(msg.get("role") or msg.get("type") or "").lower()

    content = msg.get("content")

    if role in ("user", "human"):

        return HumanMessage(content=str(content or ""))

    if role in ("assistant", "ai"):

        ai = AIMessage(content=str(content or ""))

        tc = msg.get("tool_calls")

        if tc:

            ai.tool_calls = list(tc)

        return ai

    if role == "tool":

        return ToolMessage(

            content=str(content or ""),

            tool_call_id=str(msg.get("tool_call_id") or ""),

            name=str(msg.get("name") or ""),

        )

    return None





def drop_orphan_tool_messages(messages: list[Any]) -> list[Any]:

    """去掉 orphan ToolMessage，并清掉未配齐 tool 结果的 AIMessage.tool_calls。"""

    normalized: list[BaseMessage] = []

    for raw in messages or []:

        msg = _coerce_message(raw) if not isinstance(raw, BaseMessage) else raw

        if msg is not None:

            normalized.append(msg)



    out: list[BaseMessage] = []

    pending: set[str] = set()

    for msg in normalized:

        if isinstance(msg, AIMessage):

            if pending and out and isinstance(out[-1], AIMessage):

                out[-1] = _ai_without_tool_calls(out[-1])

            pending = _call_ids(msg)

            out.append(msg)

            continue

        if isinstance(msg, ToolMessage):

            tid = str(getattr(msg, "tool_call_id", "") or "")

            if tid and tid in pending:

                out.append(msg)

                pending.discard(tid)

            continue

        if pending and out and isinstance(out[-1], AIMessage):

            out[-1] = _ai_without_tool_calls(out[-1])

        pending = set()

        out.append(msg)



    if pending and out and isinstance(out[-1], AIMessage):

        out[-1] = _ai_without_tool_calls(out[-1])

    return out





def clear_agent_checkpoint_thread(thread_id: str) -> None:

    from agent.agent_checkpointer import get_agent_checkpointer



    tid = (thread_id or "").strip()

    if not tid:

        return

    cp = get_agent_checkpointer()

    try:

        if hasattr(cp, "delete_thread"):

            cp.delete_thread(tid)

    except Exception:

        return





class CompanionToolPairMiddleware(AgentMiddleware):

    def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:

        messages = list(state.get("messages") or [])

        if not messages:

            return None

        packed = drop_orphan_tool_messages(messages)

        if len(packed) == len(messages) and packed == messages:

            return None

        from middleware.middleware_trace import append_middleware_trace

        append_middleware_trace(
            "CompanionToolPairMiddleware",
            f"清理孤儿 ToolMessage：{len(messages)} → {len(packed)} 条",
            runtime=runtime,
        )
        # messages 通道是追加合并；只返回 packed 会把孤儿 ToolMessage 留在状态里，下一跳模型仍 400
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *packed]}


