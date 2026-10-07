"""子 Agent 注册表：主 Agent 委派 → SubAgentTask → SubAgent.run → SubAgentResult。"""



from __future__ import annotations



from collections.abc import Callable

from dataclasses import dataclass

from typing import Any



from ...schemas.agent_protocol import (

    SubAgent,

    SubAgentConclusion,

    SubAgentDelegation,

    SubAgentId,

    SubAgentResult,

    SubAgentTask,

)

from ..schemas.relevance import RelevanceResult

from ..schemas.symptom import SymptomCheckResult

from .sub_agents.disease_retrieval import DiseaseRetrievalSubAgent, create_disease_retrieval_agent

from .sub_agents.symptom_match import SymptomMatchSubAgent, create_symptom_match_agent



__all__ = [

    "SubAgentRegistryEntry",

    "SUB_AGENT_REGISTRY",

    "format_sub_agents_for_prompt",

    "get_sub_agent_entry",

    "list_sub_agent_payload",

    "list_sub_agent_summaries",

    "run_sub_agent",

]



_DISEASE_AGENT = create_disease_retrieval_agent()

_SYMPTOM_AGENT = create_symptom_match_agent()





@dataclass(frozen=True, slots=True)

class SubAgentRegistryEntry:

    """子 Agent 元数据 + 协议实现。"""



    id: SubAgentId

    title: str

    summary: str

    phases: tuple[int, ...]

    tools: tuple[str, ...]

    responsibilities: tuple[str, ...]

    agent: SubAgent



    def to_summary_dict(self) -> dict[str, Any]:

        return {

            "id": self.id,

            "title": self.title,

            "summary": self.summary,

            "phases": list(self.phases),

            "tools": list(self.tools),

            "responsibilities": list(self.responsibilities),

        }



    def to_detail_dict(self) -> dict[str, Any]:

        return {

            **self.to_summary_dict(),

            "protocol": "SubAgent（name + run(task, emit) -> SubAgentResult）",

            "delegation_hint": (

                f"主 Agent 在阶段 {list(self.phases)} 且需检索/匹配时，"

                f"委派 sub_agent_id={self.id}，tools_hint={list(self.tools)}"

            ),

        }





SUB_AGENT_REGISTRY: tuple[SubAgentRegistryEntry, ...] = (

    SubAgentRegistryEntry(

        id="disease_retrieval",

        title="疾病检索子 Agent",

        summary="第一大轮：在已锁定部位下，用语义检索匹配候选疾病。",

        phases=(1,),

        tools=("search_diseases_by_symptom", "list_diseases_for_body_part"),

        responsibilities=(

            "接收 SubAgentTask.instruction 自包含执行",

            "调用 Milvus RAG 检索疾病",

            "交回 SubAgentResult.done + disease_suggested",

        ),

        agent=_DISEASE_AGENT,

    ),

    SubAgentRegistryEntry(

        id="symptom_match",

        title="症状匹配子 Agent",

        summary="第二大轮：在已锁定疾病内，将口语描述映射到官方症状名。",

        phases=(2,),

        tools=("lookup_disease_symptom_catalog",),

        responsibilities=(

            "拉取疾病官方症状表",

            "判断 matched / candidate_symptoms",

            "模糊时 SubAgentResult.need_input 挂起追问",

        ),

        agent=_SYMPTOM_AGENT,

    ),

)





def get_sub_agent_entry(sub_agent_id: str) -> SubAgentRegistryEntry | None:

    for entry in SUB_AGENT_REGISTRY:

        if entry.id == sub_agent_id:

            return entry

    return None





def list_sub_agent_summaries() -> list[dict[str, Any]]:

    return [entry.to_summary_dict() for entry in SUB_AGENT_REGISTRY]





def list_sub_agent_payload() -> list[dict[str, Any]]:

    return [entry.to_detail_dict() for entry in SUB_AGENT_REGISTRY]





def format_sub_agents_for_prompt(major_round: int) -> str:

    lines: list[str] = []

    for entry in SUB_AGENT_REGISTRY:

        if major_round not in entry.phases:

            continue

        tools = "、".join(entry.tools)

        lines.append(

            f"- **{entry.id}**（{entry.title}）：{entry.summary}\n"

            f"  建议 tools_hint：{tools}"

        )

    return "\n".join(lines) if lines else "（当前阶段无可委派子 Agent）"





def _run_agent_with_raw(

    entry: SubAgentRegistryEntry,

    task: SubAgentTask,

    *,

    emit: Callable[[dict[str, Any]], None] | None = None,

) -> tuple[SubAgentResult, RelevanceResult | SymptomCheckResult | None]:

    agent = entry.agent

    if isinstance(agent, DiseaseRetrievalSubAgent):

        return agent.run_with_raw(task, emit=emit)

    if isinstance(agent, SymptomMatchSubAgent):

        return agent.run_with_raw(task, emit=emit)

    result = agent.run(task, emit=emit)

    return result, None





def run_sub_agent(

    delegation: SubAgentDelegation,

    task: SubAgentTask,

    *,

    emit: Callable[[dict[str, Any]], None] | None = None,

) -> tuple[SubAgentResult, SubAgentConclusion, RelevanceResult | SymptomCheckResult | None]:

    """按任务单运行子 Agent，返回结果单 + 兼容结论 + 原始结果（tool_trace）。"""

    sub_id = delegation.sub_agent_id

    if sub_id == "none":

        skipped = SubAgentResult.done("无需子 Agent")

        return (

            skipped,

            SubAgentConclusion.from_result("none", skipped),

            None,

        )



    entry = get_sub_agent_entry(sub_id)

    if entry is None:

        failed = SubAgentResult.failed(f"未知子 Agent：{sub_id}")

        return failed, SubAgentConclusion.from_result(sub_id, failed), None



    result, raw = _run_agent_with_raw(entry, task, emit=emit)

    conclusion = SubAgentConclusion.from_result(sub_id, result)

    return result, conclusion, raw


