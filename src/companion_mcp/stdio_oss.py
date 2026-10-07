"""stdio MCP：OSS 预签名 PUT（报告上传）。"""

from __future__ import annotations

import json

import companion_mcp._bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

from algorithm.aliyun_oss_client import oss_presign_put

mcp = FastMCP("aliyun-oss")


@mcp.tool()
def oss_presign_upload(filename: str = "report.pdf", content_type: str = "application/pdf") -> str:
    """生成 OSS 预签名上传 URL，供客户端直传 PDF/图片。"""
    return json.dumps(oss_presign_put(filename, content_type=content_type), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
