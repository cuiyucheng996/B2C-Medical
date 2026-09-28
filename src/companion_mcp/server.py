"""把 C 端陪聊业务工具暴露成 MCP，供 Cursor 等客户端调用。不要在此复制算法。"""

from __future__ import annotations

import sys
from pathlib import Path

B2C_SRC = Path(__file__).resolve().parents[1]
UVGRAPH_ROOT = B2C_SRC.parent.parent
for item in (str(UVGRAPH_ROOT), str(B2C_SRC)):
    if item not in sys.path:
        sys.path.insert(0, item)

from mcp.server.fastmcp import FastMCP

from algorithm.ragflow_kb import kb_status, retrieve_chunks
from tools.companion_memory import companion_user_id
from tools.companion_tools import (
    compare_lab_history,
    companion_turn,
    explain_mentioned_report,
    explain_uploaded_report,
    lookup_life_facts,
    lookup_stated_consult,
    note_task_appointment,
    note_task_checkin,
    note_task_medication,
    note_task_remind,
    note_task_report,
    query_consult_graph,
    safety_policy_hint,
    search_exam_handbook,
    search_chroma_knowledge,
)

mcp = FastMCP("b2c-companion")


def _bind_user(user_id: str, handler: str = "", query: str = "") -> tuple[object | None, object]:
    uid = (user_id or "").strip()
    group_tok = companion_user_id.set(uid) if uid else None
    turn_tok = companion_turn.set(
        {
            "user_id": uid,
            "handler": handler,
            "query": query,
            "mentions": {},
            "alignments": [],
        }
    )
    return group_tok, turn_tok


def _reset(group_tok: object | None, turn_tok: object) -> None:
    companion_turn.reset(turn_tok)
    if group_tok is not None:
        companion_user_id.reset(group_tok)


@mcp.tool()
def b2c_query_consult_graph(handler: str = "", query: str = "", user_id: str = "") -> str:
    """只查 Neo4j 问诊图。"""
    group_tok, turn_tok = _bind_user(user_id, handler=handler, query=query)
    try:
        return query_consult_graph.invoke({"query": query})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_search_exam_handbook(query: str = "", handler: str = "", user_id: str = "") -> str:
    """只查 RAGFlow / 本地检查手册。"""
    group_tok, turn_tok = _bind_user(user_id, handler=handler, query=query)
    try:
        return search_exam_handbook.invoke({"query": query})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_search_chroma_knowledge(query: str = "", user_id: str = "") -> str:
    """只查 Chroma 疾病/症状向量。"""
    group_tok, turn_tok = _bind_user(user_id, query=query)
    try:
        return search_chroma_knowledge.invoke({"query": query})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_lookup_stated_consult(user_id: str) -> str:
    """用户曾提到的部位/疾病/症状。"""
    group_tok, turn_tok = _bind_user(user_id)
    try:
        return lookup_stated_consult.invoke({})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_lookup_life_facts(user_id: str) -> str:
    """用户仍有效的心情/事件等生活事实。"""
    group_tok, turn_tok = _bind_user(user_id)
    try:
        return lookup_life_facts.invoke({})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_note_task_remind(when: str = "") -> str:
    """记下复查/随访提醒意向（经 n8n 定时发提醒）。"""
    return note_task_remind.invoke({"when": when})


@mcp.tool()
def b2c_note_task_checkin(what: str = "") -> str:
    """记下打卡意向。"""
    return note_task_checkin.invoke({"what": what})


@mcp.tool()
def b2c_note_task_medication(drug: str = "", when: str = "") -> str:
    """记下服药提醒意向（药名 + 时间）。"""
    return note_task_medication.invoke({"drug": drug, "when": when})


@mcp.tool()
def b2c_note_task_appointment(department: str = "", when: str = "") -> str:
    """记下复诊预约意向（医院/科室 + 时间）。"""
    return note_task_appointment.invoke({"department": department, "when": when})


@mcp.tool()
def b2c_note_task_report(exam: str = "", when: str = "") -> str:
    """记下取报告提醒意向（检查项 + 时间）。"""
    return note_task_report.invoke({"exam": exam, "when": when})


@mcp.tool()
def b2c_explain_uploaded_report(query: str = "", user_id: str = "") -> str:
    """解读本轮已上传的检查报告。"""
    group_tok, turn_tok = _bind_user(user_id, handler="report_upload", query=query)
    try:
        return explain_uploaded_report.invoke({"query": query})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_explain_mentioned_report(exam: str = "", query: str = "", user_id: str = "") -> str:
    """解读用户口头提到的检查报告。"""
    group_tok, turn_tok = _bind_user(user_id, handler="report_explain", query=query)
    try:
        return explain_mentioned_report.invoke({"exam": exam})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_compare_lab_history(metric: str = "", query: str = "", user_id: str = "") -> str:
    """对比某指标与历史记录。"""
    group_tok, turn_tok = _bind_user(user_id, handler="report_compare", query=query)
    try:
        return compare_lab_history.invoke({"metric": metric})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_safety_policy(handler: str = "risk_emergency") -> str:
    """急症或剂量拦截口径。"""
    group_tok, turn_tok = _bind_user("", handler=handler)
    try:
        return safety_policy_hint.invoke({})
    finally:
        _reset(group_tok, turn_tok)


@mcp.tool()
def b2c_retrieve_exam_kb(question: str) -> str:
    """检索本地/RAGFlow 检查手册分块。"""
    hits = retrieve_chunks(question)
    status = kb_status()
    import json

    return json.dumps({"hits": hits, "kb": status.get("local_documents")}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
