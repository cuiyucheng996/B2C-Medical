"""主 Agent：听懂话 → 委派子 Agent → 组织人话答复（不直接调 Tool）。"""



from __future__ import annotations



import logging

from functools import lru_cache

from pathlib import Path

from typing import Any, Callable, Mapping



from langchain_core.messages import HumanMessage



from ...algorithm.artifact_store import format_artifact_summaries, merge_refs

from ...algorithm.tool_trace import summarize_tool_trace

from ...schemas.agent_protocol import (

    Artifact,

    MainAgentPlan,

    SubAgentConclusion,

    SubAgentDelegation,

    SubAgentResult,

    consult_fields_from_result,

)

from ..context.main_agent import (

    MainAgentRunContext,

    build_main_agent_context,

    build_main_agent_messages,

    render_main_agent_system_prompt,

)

from ..schemas.main_agent import MainAgentComposeOutput, MainAgentOutput, MainAgentRunResult

from ..schemas.relevance import RelevanceResult

from ..schemas.symptom import SymptomCheckResult

from .agent_stream import run_structured_agent_invoke

from .sub_agent_registry import list_sub_agent_payload, run_sub_agent

from .sub_agents.task_state import build_sub_agent_task



logger = logging.getLogger(__name__)



_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompt" / "main_agent_system_prompt.txt"

_COMPOSE_PROMPT = """\

你是医疗问诊主 Agent 的**最终答复**环节。主 Agent 已完成意图理解与委派，子 Agent 已交回结果单。



请根据【计划】【子 Agent 结果】【产物摘要】组织面向用户的专业、亲和人话，写入 `assistant_reply`。

- 不要暴露 Tool 名、JSON、内部字段名、artifact ref。

- 不得捏造未在结果或上下文中出现的疾病、症状。

- 敏感/寒暄/需追问时，礼貌引导用户回到问诊。"""



__all__ = [

    "get_main_agent_config",

    "run_main_agent",

]





def _empty_output(*, intent: str, **kwargs: Any) -> MainAgentOutput:

    defaults: dict[str, Any] = {

        "intent": intent,

        "is_relevant": False,

        "assistant_reply": "",

    }

    defaults.update(kwargs)

    return MainAgentOutput(**defaults)





def _resolve_delegation(plan: MainAgentPlan, ctx: MainAgentRunContext) -> SubAgentDelegation | None:

    delegation = plan.delegation

    if delegation is not None and delegation.sub_agent_id != "none":

        return delegation



    if not plan.is_relevant or plan.is_sensitive or plan.is_chitchat:

        return None



    if ctx.major_round == 1 and ctx.body_part:

        return SubAgentDelegation(

            sub_agent_id="disease_retrieval",

            tools_hint=["search_diseases_by_symptom"],

            task_brief=plan.intent,

        )

    if ctx.major_round == 2 and ctx.disease_name and not plan.needs_clarification:

        return SubAgentDelegation(

            sub_agent_id="symptom_match",

            tools_hint=["lookup_disease_symptom_catalog"],

            task_brief=plan.intent,

        )

    return None





def _merge_output(

    plan: MainAgentPlan,

    sub_result: SubAgentResult | None,

    assistant_reply: str,

) -> MainAgentOutput:

    fields = consult_fields_from_result(sub_result) if sub_result else {}

    needs_clarification = plan.needs_clarification

    if sub_result is not None and sub_result.status == "need_input":

        needs_clarification = True



    return MainAgentOutput(

        intent=plan.intent,

        is_relevant=plan.is_relevant,

        is_chitchat=plan.is_chitchat,

        is_sensitive=plan.is_sensitive,

        needs_clarification=needs_clarification,

        reason=plan.reason,

        body_part=plan.body_part,

        extracted_symptoms=list(fields.get("extracted_symptoms") or []),

        disease_suggested=list(fields.get("disease_suggested") or []),

        matched=bool(fields.get("matched")),

        confirmed_symptom=fields.get("confirmed_symptom"),

        candidate_symptoms=list(fields.get("candidate_symptoms") or []),

        normalized_description=str(fields.get("normalized_description") or ""),

        assistant_reply=assistant_reply,

    )





def _compose_reply(

    plan: MainAgentPlan,

    sub_result: SubAgentResult | None,

    ctx: MainAgentRunContext,

    *,

    emit: Callable[[dict[str, Any]], None] | None = None,

) -> str:

    if plan.is_sensitive:

        return "抱歉，我无法回答这类问题。请说说您哪里不舒服，或从右侧选择对应部位。"

    if plan.is_chitchat or not plan.is_relevant:

        return "我是医疗问诊助手，请告诉我您身体不适的部位或症状，我会帮您梳理。"



    if sub_result is not None and sub_result.status == "need_input" and sub_result.question:

        return sub_result.question

    if plan.needs_clarification:

        return "能再具体描述一下吗？例如哪里不舒服、有什么感觉、持续多久。"

    if sub_result is not None and sub_result.status == "failed":

        return sub_result.error or "抱歉，处理时遇到问题，请换个说法再试一次。"



    result_text = sub_result.output if sub_result and sub_result.output else plan.reason

    artifact_block = format_artifact_summaries(list(sub_result.artifacts if sub_result else []))

    artifact_section = artifact_block or "【产物摘要】\n（无）"

    user_block = (

        f"【近期对话】\n{ctx.recent_dialogue or '（无）'}\n\n"

        f"【本轮用户输入】\n{ctx.query or '（空）'}\n\n"

        f"【主 Agent 计划】\n"

        f"意图：{plan.intent}\n"

        f"理由：{plan.reason}\n\n"

        f"【子 Agent 结果】\n{result_text or '（无，仅路由判定）'}\n\n"

        f"{artifact_section}"

    )

    try:

        composed = run_structured_agent_invoke(

            system_prompt=_COMPOSE_PROMPT,

            agent_messages=[HumanMessage(content=user_block)],

            response_schema=MainAgentComposeOutput,

            emit=emit,

        )

        reply = composed.assistant_reply.strip()

        if reply:

            return reply

    except Exception:

        logger.exception("main agent compose failed, using fallback")



    if result_text:

        return f"好的，{result_text}。如需确认请继续说明。"

    return plan.reason or "请继续描述您的症状。"





def _tool_trace_from_raw(raw: RelevanceResult | SymptomCheckResult | None) -> list:

    if raw is None:

        return []

    return list(getattr(raw, "tool_trace", None) or [])





def _rag_context_from_raw(

    raw: RelevanceResult | SymptomCheckResult | None,

    tool_trace: list,

) -> list[str]:

    if raw is not None:

        rag = list(getattr(raw, "rag_context", None) or [])

        if rag:

            return rag

    return summarize_tool_trace(tool_trace)





def _artifact_refs(sub_result: SubAgentResult | None) -> list[str]:

    if sub_result is None:

        return []

    return [item.ref for item in sub_result.artifacts if item.ref]





def _emit_artifacts(emit: Callable[[dict[str, Any]], None] | None, artifacts: list[Artifact]) -> None:

    if emit is None or not artifacts:

        return

    for item in artifacts:

        emit({"event": "artifact", "artifact": item.to_dict()})





@lru_cache

def get_main_agent_config() -> dict[str, Any]:

    """供后台管理系统 API 返回的静态配置。"""

    return {

        "name": "main_agent",

        "title": "医疗问诊主 Agent",

        "responsibilities": [

            "听懂话：归纳用户本轮意图",

            "选子 Agent：产出 delegation + SubAgentTask",

            "组织最终答复：根据 SubAgentResult + artifact.summary 写 assistant_reply",

        ],

        "prompt_path": str(_PROMPT_PATH),

        "protocol": {

            "task": "SubAgentTask",

            "result": "SubAgentResult",

            "delegation": "SubAgentDelegation",

            "artifact": "ref + summary + size_bytes + download_url",

        },

        "sub_agents": list_sub_agent_payload(),

    }





def run_main_agent(

    state: Mapping[str, Any],

    *,

    emit: Callable[[dict[str, Any]], None] | None = None,

) -> MainAgentRunResult:

    """主 Agent 三阶段：计划（无 Tool）→ 子 Agent（Task/Result）→ 组织答复。"""

    ctx = build_main_agent_context(state)

    major_round = ctx.major_round

    state_dict = dict(state)

    if not state_dict.get("thread_id"):

        state_dict["thread_id"] = state.get("thread_id") or "anonymous"



    if not ctx.query:

        output = _empty_output(

            intent="（空输入）",

            is_chitchat=True,

            needs_clarification=True,

            reason="未收到有效输入",

            assistant_reply="请说说您哪里不舒服，或从右侧选择对应部位。",

        )

        return MainAgentRunResult(major_round=major_round, output=output, used_llm=False)



    system_prompt = render_main_agent_system_prompt(ctx)

    plan = run_structured_agent_invoke(
        #得到MainAgentPlan
        system_prompt=system_prompt,

        agent_messages=build_main_agent_messages(ctx),

        response_schema=MainAgentPlan,

        emit=emit,

    )



    if ctx.body_part and major_round == 1:

        plan.body_part = None  # type: ignore[misc]



    delegation = _resolve_delegation(plan, ctx)

    sub_result: SubAgentResult | None = None

    sub_conclusion: SubAgentConclusion | None = None

    raw: RelevanceResult | SymptomCheckResult | None = None

    artifacts: list[Artifact] = []

    sub_agent_refs = merge_refs(state_dict.get("sub_agent_refs"), [])



    if delegation is not None:

        if emit is not None:

            emit({"event": "thinking", "message": f"委派子 Agent：{delegation.sub_agent_id}…"})

        task = build_sub_agent_task(delegation, ctx, state_dict)
        #发 SubAgentTask，收 SubAgentResult
        sub_result, sub_conclusion, raw = run_sub_agent(delegation, task, emit=emit)

        artifacts = list(sub_result.artifacts or [])

        sub_agent_refs = merge_refs(sub_agent_refs, _artifact_refs(sub_result))

        _emit_artifacts(emit, artifacts)



    assistant_reply = _compose_reply(plan, sub_result, ctx, emit=emit)

    output = _merge_output(plan, sub_result, assistant_reply)



    tool_trace = _tool_trace_from_raw(raw)

    rag_context = _rag_context_from_raw(raw, tool_trace)

    if output.assistant_reply:

        rag_context = list(rag_context) + [f"主Agent答复: {output.assistant_reply}"]

    for item in artifacts:

        if item.summary:

            rag_context.append(f"artifact[{item.kind}]: {item.summary}")



    return MainAgentRunResult(

        major_round=major_round,

        output=output,

        plan=plan,

        sub_result=sub_result,

        sub_conclusion=sub_conclusion,

        artifacts=artifacts,

        sub_agent_refs=sub_agent_refs,

        tool_trace=tool_trace,

        rag_context=rag_context,

        used_llm=True,

    )


