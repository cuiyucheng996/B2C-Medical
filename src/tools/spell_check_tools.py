"""中文纠错 Agent 工具：BERT 模型 + 词典查询 + 常见错误模式检测。"""

from __future__ import annotations

import difflib
import re

from langchain_core.tools import tool

from runner.Predictor import get_spell_check_bert_predictor

# 模拟词典（生产可对接权威词典 API）
_DICTIONARY: dict[str, tuple[str, str]] = {
    "万豪": ("品牌名", "万豪国际酒店集团，非房企名称"),
    "万好万家": ("企业名", "浙江万好万家文化股份有限公司"),
    "东日": ("企业名", "浙江东日股份有限公司"),
    "冬日": ("普通词汇", "冬季的白天，非企业名称"),
}

# 常见错误模式：(正则, 建议修正, 错误类型)
_ERROR_PATTERNS: list[tuple[str, str, str]] = [
    (r"始曰苯", "使日本", "拼音输入错误"),
    (r"万豪万家", "万好万家", "企业名称混淆"),
    (r"冬日", "东日", "形近字错误（房企名称）"),
    (r"的\b(?=.*[跑跳走])", "地", "的/地混用"),
    (r"(.)\1{2,}", r"\1", "重复字符"),
    (r"在\b.*?在\b", "", "重复使用「在」字"),
]


def _format_char_diffs(original: str, corrected: str) -> str:
    if original == corrected:
        return "无字符级变更"
    lines: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, original, corrected
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            lines.append(
                f"- 位置{i1}: {original[i1:i2]!r} → {corrected[j1:j2]!r}"
            )
        elif tag == "delete":
            lines.append(f"- 位置{i1}: 删除 {original[i1:i2]!r}")
        elif tag == "insert":
            lines.append(f"- 位置{i1}: 插入 {corrected[j1:j2]!r}")
    return "\n".join(lines) if lines else "无字符级变更"


@tool
def bert_spell_correct(text: str) -> str:
    """使用训练好的 BERT 拼写纠错模型对文本做逐字纠错预测。

    适用于拼写错误、形近字、拼音输入错误等字符级问题。
    返回模型纠错结果及与原文的差异，作为 Agent 纠错基线。
    """
    corrected = get_spell_check_bert_predictor().predict(text)
    changed = text != corrected
    diff = _format_char_diffs(text, corrected)
    status = "有变更" if changed else "无变更"
    return (
        f"原文: {text}\n"
        f"BERT 纠错: {corrected}\n"
        f"状态: {status}\n"
        f"差异:\n{diff}"
    )


@tool
def query_dictionary(word: str) -> str:
    """查询中文词典获取词语的正确用法、近义词和常见搭配。

    适用于验证疑似错误词汇是否为真实存在的词语。
    """
    result = _DICTIONARY.get(word, ("未知词汇", "建议结合上下文判断"))
    return f"词条: {word}\n类型: {result[0]}\n说明: {result[1]}"


@tool
def check_common_errors(text: str) -> str:
    """检查文本中常见的中文输入错误模式（如拼音相似错误、形近字错误）。"""
    matches: list[str] = []
    for pattern, correction, error_type in _ERROR_PATTERNS:
        if re.search(pattern, text):
            matches.append(f"- '{pattern}' → '{correction}' ({error_type})")
    if matches:
        return "检测到潜在错误模式:\n" + "\n".join(matches)
    return "未检测到常见错误模式"
