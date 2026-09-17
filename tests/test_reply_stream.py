"""reply_stream 增量抽取单测。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.reply_stream import ReplyTextExtractor


def _stream(fragments: list[str]) -> tuple[str, list[str]]:
    """模拟逐块喂入，返回 (拼接的结果, 每次 feed 的增量)。"""
    ex = ReplyTextExtractor()
    deltas = [ex.feed_args(f) for f in fragments]
    return ex.text, [d for d in deltas if d]


class ReplyTextExtractorTest(unittest.TestCase):
    def test_single_chunk(self) -> None:
        text, deltas = _stream(['{"assistant_reply": "你好，我在。"}'])
        self.assertEqual(text, "你好，我在。")
        self.assertEqual(deltas, ["你好，我在。"])

    def test_char_by_char(self) -> None:
        payload = '{"assistant_reply": "你好，今天感觉怎么样？"}'
        text, deltas = _stream(list(payload))
        self.assertEqual(text, "你好，今天感觉怎么样？")
        # 逐字喂入时累加起来应等于整段
        self.assertEqual("".join(deltas), "你好，今天感觉怎么样？")

    def test_key_split_across_chunks(self) -> None:
        text, _ = _stream(['{"assist', 'ant_reply"', ': ', '"好', '的"}'])
        self.assertEqual(text, "好的")

    def test_escaped_newline(self) -> None:
        text, deltas = _stream(['{"assistant_reply": "第一行', '\\n第二行"}'])
        self.assertEqual(text, "第一行\n第二行")
        self.assertEqual("".join(deltas), "第一行\n第二行")

    def test_escape_split_across_chunks(self) -> None:
        """反斜杠和 n 分在两块里，不能提前吐出半个转义。"""
        ex = ReplyTextExtractor()
        first = ex.feed_args('{"assistant_reply": "甲\\')
        self.assertEqual(first, "甲")
        second = ex.feed_args('n乙"}')
        self.assertEqual(second, "\n乙")

    def test_unicode_escape_split(self) -> None:
        ex = ReplyTextExtractor()
        first = ex.feed_args('{"assistant_reply": "\\u4f')
        self.assertEqual(first, "")  # \u4f 还差一位十六进制，不能提前发
        second = ex.feed_args('60好"}')
        self.assertEqual(second, "你好")  # 补齐后一次吐出
        self.assertEqual(ex.text, "你好")

    def test_escaped_quote_not_treated_as_end(self) -> None:
        text, _ = _stream(['{"assistant_reply": "他说\\"好\\"了"}'])
        self.assertEqual(text, '他说"好"了')

    def test_extra_keys_ignored(self) -> None:
        text, _ = _stream(['{"other": "噪音", "assistant_reply": "正文"}'])
        self.assertEqual(text, "正文")

    def test_reset_clears_buffer(self) -> None:
        ex = ReplyTextExtractor()
        ex.feed_args('{"assistant_reply": "旧的"}')
        ex.reset()
        self.assertEqual(ex.text, "")
        self.assertEqual(ex.feed_args('{"assistant_reply": "新的"}'), "新的")

    def test_empty_feed(self) -> None:
        ex = ReplyTextExtractor()
        self.assertEqual(ex.feed_args(""), "")
        self.assertEqual(ex.text, "")

    def test_json_parses_to_same_text(self) -> None:
        """抽出来的文本必须与直接 json.loads 一致。"""
        import json

        payload = '{"assistant_reply": "复查提醒设好了。\\n时间：下周三 09:00"}'
        text, _ = _stream(list(payload))
        self.assertEqual(text, json.loads(payload)["assistant_reply"])


if __name__ == "__main__":
    unittest.main()
