"""白名单内的静态页正文抽取（与 knowledge_pipe 思路一致，供本服务独立运行）。"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        piece = data.strip()
        if piece:
            self.parts.append(piece)


def fetch_html_raw(
    url: str,
    *,
    max_bytes: int,
    timeout_sec: float = 25.0,
    user_agent: str = "b2c-directed-crawl/1.0",
) -> tuple[bytes, str]:
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("url 必须以 http:// 或 https:// 开头")
    req = Request(url, headers={"User-Agent": user_agent}, method="GET")
    with urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read(max_bytes + 1)
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw, charset


def extract_text_from_html(
    html: str,
    *,
    max_bytes_hint: int,
    max_output_chars: int = 12_000,
) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(parser.parts)).strip()
    if not text:
        raise RuntimeError("网页没有抓到正文")
    if len(html.encode("utf-8", errors="replace")) > max_bytes_hint:
        text = text[: max_bytes_hint // 4] + "\n…(截断)"
    return text[:max_output_chars]


def fetch_page_text(
    url: str,
    *,
    max_bytes: int,
    user_agent: str = "b2c-directed-crawl/1.0",
    timeout_sec: float = 25.0,
    max_output_chars: int = 12_000,
) -> str:
    raw, charset = fetch_html_raw(
        url, max_bytes=max_bytes, timeout_sec=timeout_sec, user_agent=user_agent
    )
    html = raw.decode(charset, errors="replace")
    return extract_text_from_html(
        html, max_bytes_hint=max_bytes, max_output_chars=max_output_chars
    )
