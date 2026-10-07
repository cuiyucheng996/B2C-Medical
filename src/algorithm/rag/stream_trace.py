"""RAG 管线步骤流式推到调试台（需 bind_stream_debug_emitter）。"""

from __future__ import annotations

from typing import Any

__all__ = ["rag_flow_step", "rag_flow_title"]


def rag_flow_title(channel: str, handler: str, *, tool: str = "") -> str:
    ch = (channel or "").strip() or "rag"
    h = (handler or "").strip()
    t = (tool or "").strip()
    bits = ["RAG增强"]
    if t:
        bits.append(t)
    else:
        bits.append(ch)
    if h:
        bits.append(h)
    return " · ".join(bits)


def rag_flow_step(title: str, section: str, detail: str) -> None:
    try:
        from algorithm.stream_debug import emit_rag_flow

        emit_rag_flow(title, section, detail)
    except Exception:
        pass
