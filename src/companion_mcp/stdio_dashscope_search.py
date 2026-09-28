"""stdio MCP：通义 enable_search 联网摘要。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

B2C_SRC = Path(__file__).resolve().parents[1]
UVGRAPH_ROOT = B2C_SRC.parent.parent
for item in (str(UVGRAPH_ROOT), str(B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from mcp.server.fastmcp import FastMCP

from algorithm.dashscope_web_search import dashscope_web_search

mcp = FastMCP("dashscope-search")


@mcp.tool()
def web_search(query: str) -> str:
    """联网检索公开信息，仅作科普/政策/机构信息参考，不作诊断。"""
    return json.dumps(dashscope_web_search(query), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
