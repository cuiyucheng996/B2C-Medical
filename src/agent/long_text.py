"""长文本纠错：按语义边界分块、逐块纠错、合并结果与错误位置。"""

from __future__ import annotations

import re
from typing import List

from schema.models import ChunkCorrection, CorrectionResult, ErrorDetail

__all__ = [
    "DEFAULT_MAX_CHUNK_LENGTH",
    "merge_corrections",
    "split_long_text",
]

DEFAULT_MAX_CHUNK_LENGTH = 300


def split_long_text(text: str, max_length: int = DEFAULT_MAX_CHUNK_LENGTH) -> List[str]:
    """将长文本智能分割为语义连贯的块。

    优先按句子边界（。！？；）切分，避免固定长度切断语义。
    max_length 默认 300：局部上下文足够判断错误，并留余量应对 token 计数差异。
    """
    if len(text) <= max_length:
        return [text]

    # 优先按语义边界分割（。！？；）
    sentences = re.split(r"(?<=[。！？；])", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if len(chunks) > 1 else [text]


def _normalize_error(error: ErrorDetail | dict) -> ErrorDetail:
    if isinstance(error, ErrorDetail):
        return error
    return ErrorDetail.model_validate(error)


def merge_corrections(
    chunks: List[str],
    results: List[CorrectionResult],
) -> CorrectionResult:
    """合并分块纠错结果，按原始块长度累加 offset 修正错误位置。"""
    if len(chunks) != len(results):
        raise ValueError(f"chunks 与 results 数量不一致: {len(chunks)} vs {len(results)}")

    merged_text = ""
    merged_errors: list[ErrorDetail] = []
    offset = 0

    for chunk, result in zip(chunks, results, strict=True):
        merged_text += result.corrected_text
        for error in result.errors:
            detail = _normalize_error(error)
            merged_errors.append(
                ErrorDetail(
                    position=detail.position + offset,
                    error_span=detail.error_span,
                    error_type=detail.error_type,
                    correction=detail.correction,
                    reason=detail.reason,
                )
            )
        offset += len(chunk)

    avg_confidence = sum(r.confidence for r in results) / len(results)
    explanation = (
        f"文本被分割为 {len(chunks)} 个语义块分别纠错，综合置信度: {avg_confidence:.2f}"
    )

    chunk_details = [
        ChunkCorrection(
            index=i,
            chunk_text=chunk,
            corrected_text=result.corrected_text,
            errors=[_normalize_error(e) for e in result.errors],
            confidence=result.confidence,
            explanation=result.explanation,
        )
        for i, (chunk, result) in enumerate(zip(chunks, results, strict=True))
    ]

    return CorrectionResult(
        original_text="".join(chunks),
        corrected_text=merged_text,
        errors=merged_errors,
        confidence=avg_confidence,
        explanation=explanation,
        chunks=chunk_details,
    )
