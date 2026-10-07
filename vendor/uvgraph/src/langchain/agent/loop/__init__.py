"""可注入的 Agent 循环（B/C 端共用）。"""

from .compress import compress_messages, messages_need_compression

__all__ = [
    "MAX_STEPS",
    "AgentEvent",
    "AgentLoop",
    "EventBus",
    "EventType",
    "LLMBackend",
    "LLMChunk",
    "LoopContext",
    "compress_messages",
    "messages_need_compression",
    "OpenAICompatibleBackend",
    "SessionStore",
    "ToolCallDelta",
    "run_loop_sync",
]


def __getattr__(name: str):
    if name in ("MAX_STEPS", "AgentLoop", "SessionStore", "run_loop_sync"):
        from .agent_loop import MAX_STEPS, AgentLoop, SessionStore, run_loop_sync

        return {
            "MAX_STEPS": MAX_STEPS,
            "AgentLoop": AgentLoop,
            "SessionStore": SessionStore,
            "run_loop_sync": run_loop_sync,
        }[name]
    if name in ("LLMBackend", "OpenAICompatibleBackend"):
        from .backend import LLMBackend, OpenAICompatibleBackend

        return {"LLMBackend": LLMBackend, "OpenAICompatibleBackend": OpenAICompatibleBackend}[name]
    if name in ("LLMChunk", "ToolCallDelta"):
        from .chunks import LLMChunk, ToolCallDelta

        return {"LLMChunk": LLMChunk, "ToolCallDelta": ToolCallDelta}[name]
    if name == "LoopContext":
        from .context import LoopContext

        return LoopContext
    if name in ("AgentEvent", "EventBus", "EventType"):
        from .events import AgentEvent, EventBus, EventType

        return {"AgentEvent": AgentEvent, "EventBus": EventBus, "EventType": EventType}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
