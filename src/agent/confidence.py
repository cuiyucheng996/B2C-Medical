"""纠错置信度：ROUGE-L 语义保留 + 编辑距离改动幅度。"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rouge_score import rouge_scorer as rouge_scorer_module

__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "calculate_confidence",
    "get_rouge_scorer",
    "levenshtein_distance",
]

LOW_CONFIDENCE_THRESHOLD = 0.4
_BASE_CONFIDENCE = 0.6
_ROUGE_WEIGHT = 0.3
_EDIT_RATIO_PENALTY = 0.2


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算 Levenshtein 编辑距离。"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


@lru_cache
def get_rouge_scorer() -> "rouge_scorer_module.RougeScorer":
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def calculate_confidence(original: str, corrected: str, scorer=None) -> float:
    """基于 ROUGE-L 和编辑距离计算置信度。

    confidence = 0.6 + 0.3 * rouge_l - 0.2 * edit_ratio
    """
    if original == corrected:
        return 1.0

    rouge_scorer = scorer or get_rouge_scorer()
    scores = rouge_scorer.score(original, corrected)
    rouge_l = scores["rougeL"].fmeasure

    edit_distance = levenshtein_distance(original, corrected)
    max_len = max(len(original), len(corrected))
    edit_ratio = edit_distance / max_len if max_len > 0 else 1.0

    confidence = _BASE_CONFIDENCE + _ROUGE_WEIGHT * rouge_l - _EDIT_RATIO_PENALTY * edit_ratio
    return max(0.0, min(1.0, confidence))
