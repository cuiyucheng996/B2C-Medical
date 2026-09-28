"""stdio MCP：OSS 预签名 PUT（报告上传）。"""

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

from algorithm.aliyun_oss_client import oss_presign_put

mcp = FastMCP("aliyun-oss")


@mcp.tool()
def oss_presign_upload(filename: str = "report.pdf", content_type: str = "application/pdf") -> str:
    """生成 OSS 预签名上传 URL，供客户端直传 PDF/图片。"""
    return json.dumps(oss_presign_put(filename, content_type=content_type), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
