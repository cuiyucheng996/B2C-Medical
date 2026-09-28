"""工具信封回来后：第三方模型判断 usable data 能否回答本轮问题。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

__all__ = [
    "ThinkVerdict",
    "hint_from_verdict",
    "judge_tool_envelopes",
    "parse_think_json",
    "usable_envelopes",
]

Action = Literal["compose", "retry_tools", "give_up"]

ThinkVerdict = dict[str, Any]

_MAX_THINK = 2
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def usable_envelopes(messages: list[Any]) -> list[dict[str, Any]]:
    tails: list[ToolMessage] = []
    for msg in reversed(messages or []):
        if isinstance(msg, ToolMessage):
            tails.append(msg)
            continue
        break
    found: list[dict[str, Any]] = []
    for msg in reversed(tails):
        try:
            payload = json.loads(str(msg.content or ""))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("usable") is True:
            found.append(payload)
    return found


def parse_think_json(text: str) -> ThinkVerdict:
    raw = _FENCE.sub("", (text or "").strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return {"fit": True, "action": "compose", "reason": "思考结果无法解析，按可用证据组句"}
    try:
        row = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"fit": True, "action": "compose", "reason": "思考结果无法解析，按可用证据组句"}
    if not isinstance(row, dict):
        return {"fit": True, "action": "compose", "reason": "思考结果无法解析，按可用证据组句"}
    fit = bool(row.get("fit"))
    action = str(row.get("action") or "").strip()
    if action not in ("compose", "retry_tools", "give_up"):
        action = "compose" if fit else "retry_tools"
    if fit:
        action = "compose"
    return {
        "fit": fit,
        "action": action,
        "reason": str(row.get("reason") or "").strip()[:80],
    }


def hint_from_verdict(verdict: ThinkVerdict, *, think_count: int) -> str:
    reason = str(verdict.get("reason") or "").strip() or "证据与问题不完全对应"
    action = str(verdict.get("action") or "compose")
    if action == "compose" or verdict.get("fit"):
        return "【思考】证据与问题匹配。请只用 usable=true 的信封 data 组句，不要再调工具。"
    if action == "give_up" or think_count >= _MAX_THINK:
        return (
            f"【思考】证据不能回答该问题（{reason}）。停止调用工具，如实告诉用户对不上，不要编造。"
        )
    return (
        f"【思考】usable 的 data 不适合回答该问题（{reason}）。"
        "请换参数或改用本轮已绑定的其他工具再调一次；不要用这份 data 编造。"
    )


def _think_cfg() -> dict[str, Any]:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs import get_app_settings, get_yaml_config

    settings = get_app_settings()
    yaml_cfg = get_yaml_config() or {}
    llm = yaml_cfg.get("llm") if isinstance(yaml_cfg.get("llm"), dict) else {}
    block = llm.get("think") if isinstance(llm.get("think"), dict) else {}
    enabled = bool(getattr(settings, "thinking_enabled", True))
    if block.get("enabled") is False:
        enabled = False
    return {
        "enabled": enabled,
        "model": str(getattr(settings, "thinking_model", None) or block.get("model") or "qwen-plus"),
        "base_url": str(
            getattr(settings, "thinking_base_url", None)
            or block.get("base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "max_tokens": int(block.get("max_tokens") or 400),
        "api_key": (
            str(getattr(settings, "thinking_api_key", "") or "").strip()
            or str(getattr(settings, "dashscope_api_key", "") or "").strip()
            or str(getattr(settings, "deepseek_api_key", "") or "").strip()
        ),
    }


def _think_llm() -> Any:
    cfg = _think_cfg()
    if not cfg["enabled"] or not cfg["api_key"]:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        temperature=0,
        max_tokens=cfg["max_tokens"],
    )


def judge_tool_envelopes(
    *,
    query: str,
    major: str = "",
    intent: str = "",
    envelopes: list[dict[str, Any]],
) -> ThinkVerdict | None:
    if not envelopes:
        return None
    llm = _think_llm()
    if llm is None:
        return None
    from prompt import load_prompt

    blob = json.dumps(envelopes, ensure_ascii=False)[:4000]
    human = (
        f"细意图：{major} / {intent}\n用户问题：{query or '（空）'}\n"
        f"usable=true 的信封：\n{blob}"
    )
    try:
        from algorithm.llm_trace import STAGE_THINK, record_llm_call, serialize_langchain_messages

        msgs = [SystemMessage(content=load_prompt("companion_think")), HumanMessage(content=human)]
        resp = llm.invoke(msgs)
        text = str(getattr(resp, "content", "") or "").strip()
        record_llm_call(
            STAGE_THINK,
            serialize_langchain_messages(msgs),
            response=text,
            label="judge_tool_envelopes",
        )
    except Exception:
        logger.warning("tool think llm failed", exc_info=True)
        return None
    return parse_think_json(text)
