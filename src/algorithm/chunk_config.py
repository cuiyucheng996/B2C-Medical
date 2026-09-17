"""手册入库分块参数：yaml 默认 + 进程内覆盖，评测页可调。"""

from __future__ import annotations

from typing import Any

__all__ = ["get_chunk_config", "set_chunk_config"]

_OVERRIDE: dict[str, Any] = {}


def _as_delim(value: Any) -> str:
    if value is None:
        return "\n"
    text = str(value).replace("\\n", "\n")
    return text or "\n"

_INT_KEYS = frozenset({"chunk_token_num", "local_max_chars", "local_naive_chars", "local_overlap"})
_STR_KEYS = frozenset(
    {
        "embedding_provider",
        "embedding_model",
        "ragflow_embedding_model",
        "chunk_method",
        "delimiter",
        "layout_recognize",
    }
)


def _defaults() -> dict[str, Any]:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs import RAGFLOW_CONFIG, get_app_settings

    settings = get_app_settings()
    rf = RAGFLOW_CONFIG if isinstance(RAGFLOW_CONFIG, dict) else {}
    return {
        "embedding_provider": str(getattr(settings, "embedding_provider", "") or "huggingface"),
        "embedding_model": str(getattr(settings, "embedding_model", "") or "BAAI/bge-small-zh-v1.5"),
        "ragflow_embedding_model": str(rf.get("embedding_model") or "text-embedding-v3@Tongyi-Qianwen"),
        "chunk_method": str(rf.get("chunk_method") or "naive"),
        "chunk_token_num": int(rf.get("chunk_token_num") or 256),
        "delimiter": _as_delim(rf.get("delimiter")),
        "layout_recognize": str(rf.get("layout_recognize") or "Plain Text"),
        "local_max_chars": int(rf.get("local_max_chars") or 1400),
        "local_naive_chars": int(rf.get("local_naive_chars") or 500),
        "local_overlap": int(rf.get("local_overlap") or 60),
    }


def get_chunk_config() -> dict[str, Any]:
    row = _defaults()
    row.update(_OVERRIDE)
    if row.get("delimiter") == "\\n":
        row["delimiter"] = "\n"
    return row


def set_chunk_config(patch: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(patch or {})
    for key in _INT_KEYS:
        if key not in data or data[key] is None or data[key] == "":
            continue
        value = int(data[key])
        if value < 1 or value > 8000:
            raise ValueError(f"{key} 超出范围")
        _OVERRIDE[key] = value
    for key in _STR_KEYS:
        if key not in data or data[key] is None:
            continue
        text = str(data[key]).strip()
        if key == "delimiter":
            text = text.replace("\\n", "\n") or "\n"
        if not text:
            continue
        _OVERRIDE[key] = text
    return get_chunk_config()
