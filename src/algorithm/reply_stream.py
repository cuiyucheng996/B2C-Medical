"""从 CompanionReply 的流式 tool args 里增量抽出 assistant_reply 文本。

结构化输出走 tool calling，模型流出来的是 JSON 片段而非纯文本：
    {'"', 'assistant', '_reply', '": "', '你好', ...}
本模块边收边解码，只吐 assistant_reply 的值，转义未收完时先不发。
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["ReplyTextExtractor"]

# 值开始位置：JSON 里 assistant_reply 这个键之后的引号
_KEY_RE = re.compile(r'"assistant_reply"\s*:\s*"')

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _decode_partial(raw: str) -> str:
    """解码 JSON 字符串字面量内部能确定的前缀；末尾转义不完整就停在它前面。"""
    out: list[str] = []
    i = 0
    size = len(raw)
    while i < size:
        ch = raw[i]
        if ch == '"':
            break  # 字符串结束
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= size:
            break  # 反斜杠后面还没到，等下一块
        nxt = raw[i + 1]
        if nxt == "u":
            if i + 6 > size:
                break  # \uXXXX 没收全
            try:
                code = int(raw[i + 2 : i + 6], 16)
            except ValueError:
                break
            if 0xD800 <= code <= 0xDBFF:
                # 高代理，必须凑齐低代理才能算出字符
                if i + 12 > size or raw[i + 6 : i + 8] != "\\u":
                    break
                try:
                    low = int(raw[i + 8 : i + 12], 16)
                except ValueError:
                    break
                if not 0xDC00 <= low <= 0xDFFF:
                    break
                out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                i += 12
                continue
            out.append(chr(code))
            i += 6
            continue
        mapped = _SIMPLE_ESCAPES.get(nxt)
        out.append(nxt if mapped is None else mapped)
        i += 2
    return "".join(out)


class ReplyTextExtractor:
    """喂 tool args 片段，返回本轮新解出的文本增量。"""

    def __init__(self) -> None:
        self._raw = ""
        self._value_start = -1
        self._emitted = 0

    def feed_args(self, fragment: str) -> str:
        """收一段 tool args，返回还没发过的 assistant_reply 文本（可能为空串）。"""
        text = str(fragment or "")
        if not text:
            return ""
        self._raw += text
        if self._value_start < 0:
            hit = _KEY_RE.search(self._raw)
            if not hit:
                return ""
            self._value_start = hit.end()
        decoded = _decode_partial(self._raw[self._value_start :])
        if len(decoded) <= self._emitted:
            return ""
        delta = decoded[self._emitted :]
        self._emitted = len(decoded)
        return delta

    @property
    def text(self) -> str:
        """已解出的完整文本。"""
        if self._value_start < 0:
            return ""
        return _decode_partial(self._raw[self._value_start :])

    def reset(self) -> None:
        """新一轮模型调用：清空累积，但保留已发出的计数。"""
        self._raw = ""
        self._value_start = -1
        self._emitted = 0


def tool_call_name(chunk: Any) -> str:
    """取 AIMessageChunk 里 CompanionReply 那个 tool call 的名字（没有则空串）。"""
    for item in getattr(chunk, "tool_call_chunks", None) or []:
        name = str((item or {}).get("name") or "").strip()
        if name:
            return name
    return ""
