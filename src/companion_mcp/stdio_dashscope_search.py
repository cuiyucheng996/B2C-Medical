"""stdio MCP：通义 enable_search 联网摘要。"""

from __future__ import annotations

import json

import companion_mcp._bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

from algorithm.dashscope_web_search import dashscope_web_search

mcp = FastMCP("dashscope-search")


@mcp.tool()
def web_search(query: str) -> str:
    """联网检索公开信息，仅作科普/政策/机构信息参考，不作诊断。"""
    return json.dumps(dashscope_web_search(query), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
