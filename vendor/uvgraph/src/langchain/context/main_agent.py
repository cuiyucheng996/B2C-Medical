"""主 Agent 运行上下文与 prompt 渲染。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage

from ...algorithm.round1 import body_from_state, disease_from_state, extract_query, latest_human_text
from ...algorithm.round2 import symptoms_for_disease
from .relevance import format_recent_dialogue

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "main_agent_system_prompt.txt"

__all__ = [
    "MainAgentRunContext",
    "build_main_agent_messages",
    "build_main_agent_context",
    "render_main_agent_system_prompt",
]

_PHASE_TITLES = {
    1: "第一大轮 · 锁定部位与疾病类型",
    2: "第二大轮 · 锁定症状",
}

_OPERATION_RULES = {
    1: """\
1. 判断用户输入是否与身体不适/症状相关；寒暄或无关 → `is_relevant=false`，不委派子 Agent。
2. 与症状相关：可直接调用注册表工具，或委派 `disease_retrieval`（`tools_hint` 含 `search_diseases_by_symptom`）。
3. **尚无已选部位**：同时推断 `body_part`（八大部位之一）。
4. **已有已选部位**：`body_part` 保持 null，检索时带上该部位过滤。
5. 仅点选部位、没有症状描述时：`tools_hint` 可含 `list_diseases_for_body_part`。
6. 不要填写 `disease_suggested`（由子 Agent 返回）。""",
    2: """\
1. 用户已锁定疾病，任务是匹配官方症状名。
2. 与症状相关时委派 `symptom_match`，`tools_hint` 含 `lookup_disease_symptom_catalog`。
3. 描述过短或与症状无关 → `needs_clarification=true`，可不委派。
4. 不要填写 `matched` / `confirmed_symptom` / `candidate_symptoms`（由子 Agent 返回）。""",
}

_RED_LINES = """\
- 主模型可调用注册表中的全部工具；检索与匹配也可委派子 Agent（`disease_retrieval` / `symptom_match`）。
- 禁止编造部位、疾病、症状名。
- 禁止输出诊断结论、处方、用药建议。
- 敏感/违规内容：`is_sensitive=true`，礼貌拒绝并引导回问诊。
- 有具体不适描述时必须委派 `disease_retrieval`，不要只列部位下全部疾病。"""


@dataclass(slots=True)
class MainAgentRunContext:
    major_round: int
    query: str
    body_part: str | None
    disease_name: str | None
    symptom_options: list[str]
    recent_dialogue: str
    phase_title: str
    phase_context: str


def build_main_agent_context(state: Mapping[str, Any]) -> MainAgentRunContext:
    major_round = int(state.get("major_round") or 1)
    messages = list(state.get("messages") or [])
    query = extract_query(state) or latest_human_text(state)
    body = body_from_state(state)
    disease = disease_from_state(state)
    symptom_options = list(state.get("symptom_options") or state.get("candidate_top3") or [])
    if not symptom_options and disease:
        symptom_options = symptoms_for_disease(disease)

    if major_round >= 2:
        phase_context = (
            f"已选部位：{body or '（无）'}\n"
            f"已锁定疾病：{disease or '（缺失）'}\n"
            f"候选症状（参考）：{'、'.join(symptom_options[:12]) if symptom_options else '（待 Tool 拉取）'}"
        )
    else:
        phase_context = (
            f"已选部位：{body or '（尚未选定）'}\n"
            f"已锁定疾病：{disease or '（尚未选定）'}"
        )

    return MainAgentRunContext(
        major_round=major_round,
        query=query.strip(),
        body_part=body,
        disease_name=disease,
        symptom_options=symptom_options,
        recent_dialogue=format_recent_dialogue(messages),
        phase_title=_PHASE_TITLES.get(major_round, _PHASE_TITLES[1]),
        phase_context=phase_context,
    )


def render_main_agent_system_prompt(ctx: MainAgentRunContext) -> str:
    from ..agent.sub_agent_registry import format_sub_agents_for_prompt

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    phase = ctx.major_round if ctx.major_round in _OPERATION_RULES else 1
    return template.format(
        phase_title=ctx.phase_title,
        phase_context=ctx.phase_context,
        sub_agents=format_sub_agents_for_prompt(phase),
        operation_rules=_OPERATION_RULES[phase],
        red_lines=_RED_LINES,
    )


def build_main_agent_messages(ctx: MainAgentRunContext) -> list[BaseMessage]:
    content = (
        f"【近期对话】\n{ctx.recent_dialogue or '（无）'}\n\n"
        f"【本轮用户输入】\n{ctx.query or '（空）'}"
    )
    return [HumanMessage(content=content)]
