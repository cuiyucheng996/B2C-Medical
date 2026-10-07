"""多问法粗召回后的 RRF 精排。

不同问法各自 embedding 召回一块有序列表；完全相同的块（UTF-8 hash）
在更多列表里出现、且排名越靠前，RRF 得分越高。

当前未接入 Agent。

用法（项目根目录）：
    uv run python -m src.langchain.rerank.algorithm
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_RRF_K = 60

__all__ = [
    "ContextDocument",
    "DEFAULT_RRF_K",
    "DocumentDedupStore",
    "RerankedDocument",
    "dedupe_documents",
    "document_content_hash",
    "document_to_utf8",
    "merge_retrieval_documents",
    "normalize_document_text",
    "rrf_rerank",
]


def normalize_document_text(text: str) -> str:
    """去重前统一空白，避免仅空格差异导致 hash 不同。"""
    return " ".join(text.strip().split())


def document_to_utf8(text: str) -> bytes:
    """将 document 文本编码为 UTF-8 bytes。"""
    return normalize_document_text(text).encode("utf-8")


def document_content_hash(text: str, *, algorithm: str = "sha256") -> str:
    """document → UTF-8 → hash hex。"""
    payload = document_to_utf8(text)
    try:
        digest = hashlib.new(algorithm, payload)
    except ValueError as exc:
        raise ValueError(f"不支持的 hash 算法: {algorithm}") from exc
    return digest.hexdigest()


@dataclass(slots=True)
class ContextDocument:
    """单条 context document。"""

    text: str
    content_hash: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentDedupStore:
    """基于 hash set 的 document 去重容器。"""

    def __init__(self, *, hash_algorithm: str = "sha256") -> None:
        self._hash_algorithm = hash_algorithm
        self._seen_hashes: set[str] = set()
        self._documents: list[ContextDocument] = []

    @property
    def seen_hashes(self) -> set[str]:
        return set(self._seen_hashes)

    @property
    def documents(self) -> list[ContextDocument]:
        return list(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    def has(self, text: str) -> bool:
        content_hash = document_content_hash(text, algorithm=self._hash_algorithm)
        return content_hash in self._seen_hashes

    def add(
        self,
        text: str,
        *,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextDocument | None:
        """写入 document；重复则返回 None。"""
        normalized = normalize_document_text(text)
        if not normalized:
            return None

        content_hash = document_content_hash(normalized, algorithm=self._hash_algorithm)
        if content_hash in self._seen_hashes:
            return None

        self._seen_hashes.add(content_hash)
        doc = ContextDocument(
            text=normalized,
            content_hash=content_hash,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._documents.append(doc)
        return doc

    def add_many(
        self,
        texts: Iterable[str],
        *,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> list[ContextDocument]:
        added: list[ContextDocument] = []
        for text in texts:
            doc = self.add(text, source=source, metadata=metadata)
            if doc is not None:
                added.append(doc)
        return added

    def texts(self) -> list[str]:
        return [doc.text for doc in self._documents]


def dedupe_documents(
    documents: Sequence[str],
    *,
    source: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """对 document 列表去重，保留首次出现顺序。"""
    store = DocumentDedupStore()
    store.add_many(documents, source=source, metadata=metadata)
    return store.texts()


def merge_retrieval_documents(
    *batches: Sequence[str],
    source: str = "",
) -> list[str]:
    """合并多路 embedding 检索结果并去重（按批次顺序保留首次出现）。"""
    store = DocumentDedupStore()
    for index, batch in enumerate(batches):
        batch_source = source or f"retrieval_batch_{index}"
        store.add_many(batch, source=batch_source)
    return store.texts()


@dataclass(slots=True)
class RerankedDocument:
    """RRF 精排后的文档。"""

    text: str
    content_hash: str
    rrf_score: float
    hit_count: int
    ranks: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


def rrf_rerank(
    ranked_lists: Mapping[str, Sequence[str]] | Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    hash_algorithm: str = "sha256",
) -> list[RerankedDocument]:
    """对多路粗召回做 Reciprocal Rank Fusion 精排。

    同一 UTF-8 hash 的块：出现在越多问法列表里、各列表名次越靠前，得分越高。
    单条列表内相同 hash 只取首次排名。
    """
    if k <= 0:
        raise ValueError("RRF k 必须为正整数")

    if isinstance(ranked_lists, Mapping):
        named_lists = [(name, list(items)) for name, items in ranked_lists.items()]
    else:
        named_lists = [(f"query_{index}", list(items)) for index, items in enumerate(ranked_lists)]

    scores: dict[str, float] = {}
    texts: dict[str, str] = {}
    ranks: dict[str, dict[str, int]] = {}
    sources: dict[str, list[str]] = {}

    for list_name, items in named_lists:
        seen_in_list: set[str] = set()
        for rank, raw in enumerate(items, start=1):
            normalized = normalize_document_text(raw)
            if not normalized:
                continue
            content_hash = document_content_hash(normalized, algorithm=hash_algorithm)
            if content_hash in seen_in_list:
                continue
            seen_in_list.add(content_hash)

            texts.setdefault(content_hash, normalized)
            scores[content_hash] = scores.get(content_hash, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(content_hash, {})[list_name] = rank
            sources.setdefault(content_hash, []).append(list_name)

    ranked = [
        RerankedDocument(
            text=texts[content_hash],
            content_hash=content_hash,
            rrf_score=score,
            hit_count=len(ranks[content_hash]),
            ranks=ranks[content_hash],
            sources=sources[content_hash],
        )
        for content_hash, score in scores.items()
    ]
    ranked.sort(key=lambda item: (-item.rrf_score, -item.hit_count, item.content_hash))
    return ranked


def main() -> None:
    lists = {
        "头痛睡不着": [
            "头痛：偏头痛，一侧搏动性疼痛",
            "失眠：入睡困难，多梦易醒",
            "眩晕：头晕目眩",
        ],
        "头很疼夜里醒": [
            "失眠：入睡困难，多梦易醒",
            "头痛：偏头痛，一侧搏动性疼痛",
            "心悸：心慌气短",
        ],
        "偏头痛伴失眠": [
            "头痛：偏头痛，一侧搏动性疼痛",
            "不寐：夜间易醒",
        ],
    }
    results = rrf_rerank(lists)
    print("=== RRF 精排 ===")
    for index, item in enumerate(results, start=1):
        print(
            f"{index}. hit={item.hit_count} score={item.rrf_score:.6f} "
            f"ranks={item.ranks} | {item.text}"
        )


if __name__ == "__main__":
    main()
