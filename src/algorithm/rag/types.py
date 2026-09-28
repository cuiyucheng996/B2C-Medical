"""RAG 增强管线：统一 Hit 与配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RagChannel = Literal["handbook", "chroma", "graph"]

__all__ = ["RagChannel", "RagHit", "RagPipelineConfig"]


@dataclass
class RagHit:
    text: str
    score: float = 0.0
    recall_rank: int = 0
    channel: str = ""
    doc: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "recall_rank": self.recall_rank,
            "channel": self.channel,
            "doc": self.doc,
            **self.meta,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> RagHit:
        return cls(
            text=str(row.get("text") or ""),
            score=float(row.get("score") or 0),
            recall_rank=int(row.get("recall_rank") or 0),
            channel=str(row.get("channel") or ""),
            doc=str(row.get("doc") or ""),
            meta={
                k: v
                for k, v in row.items()
                if k not in {"text", "score", "recall_rank", "channel", "doc"}
            },
        )


@dataclass
class RagPipelineConfig:
    enhanced: bool = True
    multi_query: bool = True
    max_queries: int = 3
    rrf_k: int = 60
    fetch_k: int = 12
    top_k: int = 8
    use_cache: bool = True
    cache_ttl_seconds: int = 300
    keyword_rerank: bool = True


