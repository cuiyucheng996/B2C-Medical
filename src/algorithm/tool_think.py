"""工具信封回来后：第三方模型开思考：data 对不上就提示本轮再调工具。"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from algorithm.structured_output import invoke_structured_llm, parse_json_with_schema
from prompt import load_prompt
from prompt.structured_block import compose_structured_system_prompt
from schema.llm_outputs import CompanionThinkOutput

logger = logging.getLogger(__name__)

__all__ = [
    "ThinkVerdict",
    "hint_from_verdict",
    "judge_tool_envelopes",
    "parse_think_json",
    "usable_envelopes",
    "_think_llm",
]

Action = Literal["compose", "retry_tools", "give_up"]

ThinkVerdict = dict[str, Any]

_MAX_THINK = 2

_THINK_EXAMPLES = [
    {"fit": True, "action": "compose", "reason": "data.items 已含指标参考范围"},
    {"fit": False, "action": "retry_tools", "reason": "query 太宽，需换检查名"},
    {"fit": False, "action": "give_up", "reason": "图谱无该证型，不宜编造"},
]


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


def _think_output_to_verdict(row: CompanionThinkOutput) -> ThinkVerdict:
    fit = bool(row.fit)
    action = str(row.action)
    if action not in ("compose", "retry_tools", "give_up"):
        action = "compose" if fit else "retry_tools"
    if fit:
        action = "compose"
    return {
        "fit": fit,
        "action": action,
        "reason": str(row.reason or "").strip()[:80],
    }


def parse_think_json(text: str) -> ThinkVerdict:
    try:
        return _think_output_to_verdict(parse_json_with_schema(text, CompanionThinkOutput))
    except Exception:
        return {"fit": True, "action": "compose", "reason": "思考结果无法解析，按可用证据组句"}


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


def _think_system_prompt() -> str:
    return compose_structured_system_prompt(
        load_prompt("companion_think"),
        CompanionThinkOutput,
        examples=_THINK_EXAMPLES,
    )


def _think_fallback() -> CompanionThinkOutput:
    return CompanionThinkOutput(
        fit=True,
        action="compose",
        reason="思考结构化失败，按可用证据组句",
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
    blob = json.dumps(envelopes, ensure_ascii=False)[:4000]
    human = (
        f"细意图：{major} / {intent}\n用户问题：{query or '（空）'}\n"
        f"usable=true 的信封：\n{blob}"
    )
    try:
        from algorithm.llm_trace import STAGE_THINK, record_llm_call, serialize_langchain_messages

        msgs = [SystemMessage(content=_think_system_prompt()), HumanMessage(content=human)]
        row = invoke_structured_llm(
            llm,
            CompanionThinkOutput,
            msgs,
            label="judge_tool_envelopes",
            fallback=_think_fallback,
        )
        record_llm_call(
            STAGE_THINK,
            serialize_langchain_messages(msgs),
            response=row.model_dump_json(),
            label="judge_tool_envelopes",
        )
        return _think_output_to_verdict(row)
    except Exception:
        logger.warning("tool think llm failed", exc_info=True)
        return _think_output_to_verdict(_think_fallback())
