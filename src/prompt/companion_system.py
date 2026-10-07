"""陪聊子 Agent system：可缓存静态块 + 每路动态块。"""

from __future__ import annotations

from functools import lru_cache

from prompt import load_prompt
from prompt.companion_dispatch import build_companion_route_tail, build_companion_skill_prompt
from prompt.structured_block import compose_structured_system_prompt
from schema.companion_reply import CompanionReply

__all__ = [
    "build_companion_system_dynamic",
    "build_companion_system_skill",
    "build_companion_system_static",
]


@lru_cache(maxsize=1)
def build_companion_system_static() -> str:
    """各路完全相同的 system 前缀（工具调度头 + 组句 + 结构化 Schema）。"""
    return "\n\n".join(
        [
            load_prompt("companion_tools_head"),
            load_prompt("companion_compose"),
            compose_structured_system_prompt(
                "最终必须通过 CompanionReply 工具提交 assistant_reply（见下 Schema，勿在正文输出 JSON）。",
                CompanionReply,
                examples=[{"assistant_reply": "您好，有什么可以帮您的？"}],
            ),
        ]
    )


def build_companion_system_skill(*, major: str = "", handler: str = "") -> str:
    """按 handler（或 major 回退）的 SOP；多细意图同 handler 可共享前缀缓存。"""
    return build_companion_skill_prompt(major=major, handler=handler)


def build_companion_system_dynamic(
    *,
    major: str = "",
    intent: str = "",
    handler: str = "",
    bound_tools: list[str] | None = None,
) -> str:
    """本路 intent、工具绑定与任务边界（不缓存）。"""
    parts = [
        build_companion_route_tail(
            major=major,
            intent=intent,
            handler=handler,
            bound_tools=bound_tools,
        )
    ]
    if major or intent or handler:
        parts.append(
            f"本任务只处理：{major} / {intent}（{handler}）。勿回答其它意图；并行工具结果按此甄别。"
        )
    return "\n\n".join(p for p in parts if p)
