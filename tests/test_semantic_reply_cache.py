"""语义缓存单元测试（不调用 embedding API）。"""

from __future__ import annotations

import os

import pytest

from algorithm.semantic_reply_cache import (
    _cosine,
    remember_semantic_reply,
    semantic_cache_enabled,
    should_try_semantic_cache,
)


def test_cosine_identical():
    a = [1.0, 0.0, 0.0]
    assert _cosine(a, a) == pytest.approx(1.0)


def test_should_try_semantic_cache_short_text():
    assert should_try_semantic_cache("你好呀", preprocess={}) is semantic_cache_enabled()


def test_should_skip_long_text():
    assert should_try_semantic_cache("x" * 200, preprocess={}) is False


def test_remember_skips_graph_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMPANION_SEMANTIC_CACHE", "1")

    def _no_embed(_: str):
        return [1.0, 0.0]

    monkeypatch.setattr(
        "algorithm.semantic_reply_cache._embed_query",
        lambda text: _no_embed(text),
    )
    remember_semantic_reply("u1", "你好", "嗨", [], source="graph")
    from algorithm.semantic_reply_cache import _PROCESS_FALLBACK

    assert "u1" not in _PROCESS_FALLBACK
