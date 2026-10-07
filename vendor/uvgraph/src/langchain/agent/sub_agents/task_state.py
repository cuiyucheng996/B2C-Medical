"""SubAgentTask 与 LangGraph state 互转。"""



from __future__ import annotations



from typing import Any, Mapping, Protocol



from ....algorithm.artifact_store import get_stored_artifact, resolve_refs

from ....schemas.agent_protocol import SubAgentDelegation, SubAgentTask





class _TaskContext(Protocol):

    major_round: int

    query: str

    body_part: str | None

    disease_name: str | None

    symptom_options: list[str]



__all__ = [

    "build_sub_agent_task",

    "task_to_state",

]





def _append_ref_summaries(lines: list[str], refs: list[str]) -> None:

    if not refs:

        return

    lines.append("【可引用产物（refs）】")

    for ref in refs:

        stored = get_stored_artifact(ref)

        if stored is None:

            lines.append(f"- {ref}：（句柄无效或已过期）")

            continue

        lines.append(f"- {ref}：{stored.summary}")





def build_instruction(delegation: SubAgentDelegation, ctx: _TaskContext, refs: list[str]) -> str:

    """主 Agent 侧：把委派 + 上下文展开为自包含 instruction。"""

    lines = [

        delegation.task_brief or ctx.query or "（无任务摘要）",

        f"用户本轮原文：{ctx.query or '（空）'}",

    ]

    if ctx.body_part:

        lines.append(f"用户已选定部位：{ctx.body_part}。")

    if ctx.disease_name:

        lines.append(f"已锁定疾病：{ctx.disease_name}。")

    if ctx.symptom_options:

        preview = "、".join(ctx.symptom_options[:8])

        lines.append(f"候选症状（参考）：{preview}。")

    if delegation.tools_hint:

        lines.append(f"建议优先使用工具：{', '.join(delegation.tools_hint)}。")

    _append_ref_summaries(lines, refs)

    return "\n".join(lines)





def build_sub_agent_task(

    delegation: SubAgentDelegation,

    ctx: _TaskContext,

    state: Mapping[str, Any],

) -> SubAgentTask:

    snapshot = state.get("sub_agent_snapshot")

    if isinstance(snapshot, dict):

        resume_token = state.get("sub_agent_resume_token")

    else:

        snapshot = None

        resume_token = None



    refs = list(state.get("sub_agent_refs") or [])



    return SubAgentTask(

        instruction=build_instruction(delegation, ctx, refs),

        refs=refs,

        resume_token=resume_token if isinstance(resume_token, str) else None,

        context={

            "sub_agent_id": delegation.sub_agent_id,

            "tools_hint": list(delegation.tools_hint),

            "major_round": ctx.major_round,

            "body_part": ctx.body_part,

            "disease_name": ctx.disease_name,

            "query": ctx.query,

            "thread_id": state.get("thread_id"),

            "messages": list(state.get("messages") or []),

            "selected_body_part": state.get("selected_body_part"),

            "selected_disease": state.get("selected_disease"),

            "symptom_options": list(state.get("symptom_options") or []),

            "snapshot": snapshot,

        },

    )





def task_to_state(task: SubAgentTask) -> dict[str, Any]:

    """子 Agent 侧：从任务单还原最小可执行 state。"""

    ctx = dict(task.context or {})

    snapshot = ctx.pop("snapshot", None)

    if task.resume_token and isinstance(snapshot, dict):

        ctx.update(snapshot)



    state: dict[str, Any] = {

        "major_round": int(ctx.get("major_round") or 1),

        "messages": list(ctx.get("messages") or []),

        "thread_id": ctx.get("thread_id"),

        "selected_body_part": ctx.get("selected_body_part") or ctx.get("body_part"),

        "selected_disease": ctx.get("selected_disease") or ctx.get("disease_name"),

        "symptom_options": list(ctx.get("symptom_options") or []),

        "sub_agent_refs": list(task.refs or []),

    }

    if ctx.get("body_part"):

        state["selected_body_part"] = ctx["body_part"]

    if ctx.get("disease_name"):

        state["selected_disease"] = ctx["disease_name"]



    payloads = resolve_refs(list(task.refs or []))

    if payloads:

        state["artifact_payloads"] = payloads

    return state


