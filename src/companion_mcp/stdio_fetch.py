"""stdio MCP：拉取公开网页正文。Windows 上替代 npx @modelcontextprotocol/server-fetch。"""

from __future__ import annotations

import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import companion_mcp._bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fetch")
_MAX = 80_000
_TAG = re.compile(r"<[^>]+>")


@mcp.tool()
def fetch(url: str) -> str:
    """GET 公开 http(s) 页面，返回纯文本（截断）。"""
    target = (url or "").strip()
    if not target.startswith(("http://", "https://")):
        return "只支持 http/https URL"
    req = Request(target, headers={"User-Agent": "uvgraph-b2c-fetch/1.0"})
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read(_MAX + 1)
    except HTTPError as exc:
        return f"HTTP {exc.code}"
    except URLError as exc:
        return f"请求失败: {exc.reason}"
    text = raw[:_MAX].decode("utf-8", errors="replace")
    text = _TAG.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(raw) > _MAX:
        text += " …(截断)"
    return text or "(空页面)"


if __name__ == "__main__":
    mcp.run()
