"""Embedding 模型配置与工厂。"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from ..configs import get_app_settings

__all__ = ["get_embeddings", "get_cpu_embeddings"]


@lru_cache
def get_embeddings() -> Embeddings:
    """返回项目默认 embedding 模型（单例）。"""
    cfg = get_app_settings()

    if cfg.embedding_provider == "dashscope":
        if not cfg.dashscope_api_key:
            msg = "DASHSCOPE_API_KEY is not set"
            raise ValueError(msg)
        return DashScopeEmbeddings(
            model=cfg.embedding_model,
            dashscope_api_key=cfg.dashscope_api_key,
        )

    if cfg.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        device = (os.environ.get("UVGRAPH_EMBED_DEVICE") or cfg.embedding_device or "cpu").strip()
        return HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": device},
            #归一化，使得向量长度为1，避免计算相似度时出现误差
            encode_kwargs={"normalize_embeddings": True},
        )

    msg = f"Unsupported embedding provider: {cfg.embedding_provider}"
    raise ValueError(msg)


@lru_cache
def get_cpu_embeddings() -> Embeddings:
    """问诊对齐用：始终 CPU，不与主进程 UIE 抢 GPU、不开子进程。"""
    cfg = get_app_settings()
    if cfg.embedding_provider == "dashscope":
        return get_embeddings()
    if cfg.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    msg = f"Unsupported embedding provider: {cfg.embedding_provider}"
    raise ValueError(msg)
