"""B2C vendor：勿在包初始化时拉 main_agent（缺 artifact_store 等 B2B 模块）。"""

from .stream_agent import chunk_text, sse_payload

__all__ = ["chunk_text", "sse_payload"]
