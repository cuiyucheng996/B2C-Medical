"""stdio MCP：百炼知识库 Retrieve / ListIndices。"""

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

from algorithm.bailian_kb_client import bailian_list_indices
from algorithm.bailian_kb_client import bailian_retrieve as retrieve_kb

mcp = FastMCP("bailian-kb")


@mcp.tool()
def bailian_retrieve(query: str, index_id: str = "", dense_top_k: int = 5) -> str:
    """从百炼知识库检索切片；index_id 空则用环境默认。"""
    out = retrieve_kb(
        query,
        index_id=index_id or None,
        dense_top_k=dense_top_k,
    )
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def list_indices(page: int = 1, size: int = 20) -> str:
    """列出业务空间下知识库 Index。"""
    return json.dumps(bailian_list_indices(page=page, size=size), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
