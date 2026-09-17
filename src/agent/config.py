"""纠错 Agent 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.long_text import DEFAULT_MAX_CHUNK_LENGTH

__all__ = ["SpellCheckConfig"]


@dataclass(slots=True)
class SpellCheckConfig:
    """构造 SpellCheckAgent 时可覆盖模型与行为。"""

    model: str | None = None
    temperature: float = 0.2
    use_tools: bool = True
    use_bert_tool: bool = True
    max_tokens: int = 2048
    timeout: int = 30
    max_chunk_length: int = DEFAULT_MAX_CHUNK_LENGTH
    require_high_confidence: bool = True
