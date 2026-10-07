"""从 RouterState 构建症状匹配 Agent 运行上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage

from ...algorithm.round1 import disease_from_state, extract_query, latest_human_text
from ...algorithm.round2 import normalize_colloquial_symptom, symptoms_for_disease
from .relevance import format_recent_dialogue

__all__ = [
    "SymptomRunContext",
    "build_agent_messages",
    "build_run_context",
]


@dataclass(slots=True)
class SymptomRunContext:
    """症状 Agent 运行时上下文。"""

    query: str
    disease_name: str
    body_part: str | None = None
    symptom_options: list[str] | None = None
    recent_dialogue: str = ""
    turn_index: int = 1


def build_run_context(state: Mapping[str, Any]) -> SymptomRunContext:
    messages = list(state.get("messages") or [])
    raw_query = extract_query(state) or latest_human_text(state)
    query = normalize_colloquial_symptom(raw_query)
    disease = disease_from_state(state) or ""
    options = list(state.get("symptom_options") or [])
    if not options and disease:
        options = symptoms_for_disease(disease)

    from ...algorithm.round1 import body_from_state

    return SymptomRunContext(
        query=query.strip(),
        disease_name=disease.strip(),
        body_part=body_from_state(state),
        symptom_options=options or None,
        recent_dialogue=format_recent_dialogue(messages),
        turn_index=max(1, sum(1 for m in messages if m.type == "human")),
    )


def build_agent_messages(ctx: SymptomRunContext) -> list[HumanMessage]:
    options = ctx.symptom_options or []
    option_text = "、".join(options[:12]) if options else "（暂无，请先调用 lookup_disease_symptom_catalog）"
    body_hint = f"已选部位：{ctx.body_part}" if ctx.body_part else "已选部位：无"
    content = (
        f"【会话上下文】\n"
        f"轮次：第 {ctx.turn_index} 轮用户输入\n"
        f"{body_hint}\n"
        f"已锁定疾病：{ctx.disease_name or '（缺失）'}\n\n"
        f"【近期对话】\n{ctx.recent_dialogue}\n\n"
        f"【当前疾病候选症状（参考）】\n{option_text}\n\n"
        f"【本轮用户症状描述】\n{ctx.query or '（空）'}"
    )
    return [HumanMessage(content=content)]
