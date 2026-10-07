"""陪聊上下文增量压缩：qwen 摘要新段落，旧摘要只拼接。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def _uvgraph_loop_compress():
    """只加载 compress.py，避免 import loop 包时拉起 agent_loop / tools 链。"""
    import importlib.util

    from configs.paths_b2c import VENDOR_UVGRAPH, ensure_b2c_runtime_path

    ensure_b2c_runtime_path()
    path = VENDOR_UVGRAPH / "src" / "langchain" / "agent" / "loop" / "compress.py"
    spec = importlib.util.spec_from_file_location("_uvgraph_loop_compress", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load compress from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_compress = _uvgraph_loop_compress()
KEEP_USER_TURNS = _compress.KEEP_USER_TURNS
_shrink_recent = _compress._shrink_recent
_summarize_old = _compress._summarize_old
_text = _compress._text

logger = logging.getLogger(__name__)

_OBSERVE = "【预处理观察】"

__all__ = ["apply_incremental_compress"]


def _is_observe(msg: BaseMessage) -> bool:
    return isinstance(msg, HumanMessage) and _OBSERVE in str(getattr(msg, "content", "") or "")


def _dialogue_human_indices(messages: list[BaseMessage]) -> list[int]:
    return [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, HumanMessage) and not _is_observe(msg)
    ]


def _compress_cfg() -> dict[str, Any]:
    from src.configs import get_yaml_config

    yaml_cfg = get_yaml_config() or {}
    llm = yaml_cfg.get("llm") if isinstance(yaml_cfg.get("llm"), dict) else {}
    block = llm.get("context_compress") if isinstance(llm.get("context_compress"), dict) else {}
    return block


def _summarize_delta_llm(delta: list[BaseMessage]) -> str | None:
    if not delta:
        return None
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from langchain_core.messages import HumanMessage as HM
    from langchain_core.messages import SystemMessage as SM
    from langchain_openai import ChatOpenAI
    from src.configs import get_app_settings

    from prompt import load_prompt

    settings = get_app_settings()
    api_key = (getattr(settings, "dashscope_api_key", None) or "").strip()
    if not api_key:
        return None
    cfg = _compress_cfg()
    if cfg.get("enabled") is False:
        return None
    model = str(cfg.get("model") or "qwen-plus")
    base_url = str(cfg.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    max_tokens = int(cfg.get("max_tokens") or 400)
    lines: list[str] = []
    for msg in delta:
        if _is_observe(msg):
            continue
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        if isinstance(msg, ToolMessage):
            role = str(getattr(msg, "name", None) or "工具")
        text = _text(msg).strip()
        if text:
            lines.append(f"{role}：{text[:800]}")
    if not lines:
        return None
    try:
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=max_tokens,
        )
        resp = llm.invoke(
            [
                SM(content=load_prompt("context_compress")),
                HM(content="本段新对话：\n" + "\n".join(lines)),
            ]
        )
        out = str(getattr(resp, "content", "") or "").strip()
        return out or None
    except Exception:
        logger.warning("context compress llm failed", exc_info=True)
        return None


def _merge_short_and_state(session_id: str, messages: list[BaseMessage]) -> list[BaseMessage]:
    """进模型的 messages 已含本场 history，不再用 short 全量覆盖，避免摘要和原文叠在一起。"""
    del session_id
    return list(messages)


def apply_incremental_compress(
    messages: list[BaseMessage],
    *,
    session_id: str,
) -> list[BaseMessage]:
    sid = (session_id or "").strip()
    if not messages and not sid:
        return []
    merged = _merge_short_and_state(sid, messages) if sid else list(messages)
    if not merged:
        return []
    cfg = _compress_cfg()
    keep = int(cfg.get("keep_user_turns") or KEEP_USER_TURNS)
    humans = _dialogue_human_indices(merged)
    if len(humans) <= keep:
        return [_shrink_recent(m) for m in merged]

    cut = humans[-keep]
    old, recent = merged[:cut], merged[cut:]
    old_humans = _dialogue_human_indices(old)
    covered_target = len(old_humans)

    from src.redis.context_compress import load_compress_sync, save_compress_sync

    stored = load_compress_sync(sid) if sid else {"covered_humans": 0, "segments": []}

    covered = int(stored.get("covered_humans") or 0)
    segments = list(stored.get("segments") or [])

    if covered_target > covered:
        start = old_humans[covered] if covered < len(old_humans) else cut
        delta = old[start:]
        piece = _summarize_delta_llm(delta)
        if not piece:
            piece = _summarize_old(delta)
        if piece:
            segments.append(piece)
        covered = covered_target
        if sid:
            save_compress_sync(
                sid,
                {
                    "covered_humans": covered,
                    "segments": segments,
                },
            )

    blob = "\n\n".join(segments).strip()
    head: list[BaseMessage] = []
    if blob:
        head.append(SystemMessage(content="[历史摘要]\n" + blob))
    return [*head, *[_shrink_recent(m) for m in recent]]
