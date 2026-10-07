"""Round-1 opening router schemas.

第一大轮 · 第一轮 AiThinking 之后的两路分支：
  - remind  : 用户输入与身体状态无关
  - continue: 用户输入相关，进入症状/部位解析

参考 RouterState 模式：Input → Decision → Output → 主路由状态累积。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from .consult import BodyPart, UserInputMode

# ---------------------------------------------------------------------------
# 分支标识
# ---------------------------------------------------------------------------

Round1Route = Literal["remind", "continue"]
"""AiThinking 一级路由：提醒 or 继续。"""

Round1Handler = Literal[
    "relevance_checker",  # AiThinking 本身
    "remind_agent",       # 无关 → 提醒用户
    "option_agent",       # 相关 + 用户点了选项
    "disease_selection_agent",  # 部位选定后展示疾病选项
    "infer_disease_agent",      # 从文字推断候选疾病
    "text_agent",         # 相关 + 纯自由文本
    "vague_agent",        # 相关但描述模糊，需追问
    "match_agent",        # 症状/部位匹配
    "retry_agent",        # 未命中，带记忆重试
]
"""第一大轮二级子节点 handler 标识。"""

Round2Route = Literal["remind", "continue", "clarify", "present_top3", "round3_ready"]
"""第二大轮 AiThinking 一级路由。"""

Round2Handler = Literal[
    "symptom_relevance_checker",  # AiThinking 本身
    "symptom_selection_agent",    # 展示症状选项
    "option_symptom_agent",       # 用户点选症状
    "infer_symptom_agent",        # 「没有太像的」+ 文字描述
    "text_agent",                 # 纯自由文本
    "vague_agent",                # 描述模糊
    "match_symptom_agent",        # 口语标准化 + 症状匹配
]
"""第二大轮二级子节点 handler 标识。"""

ContinueSubRoute = Literal["option", "text", "vague"]
"""continue 分支下的子路由。"""


# ---------------------------------------------------------------------------
# 子节点 Input / Output（类比 AgentInput / AgentOutput）
# ---------------------------------------------------------------------------


class BranchInput(TypedDict):
    """每个子节点的输入。"""

    query: str
    """用户原始输入文本。"""

    input_mode: UserInputMode
    """option / text / number。"""

    selected_body_part: BodyPart | None
    """用户通过 UI 选择的部位（若有）。"""

    retry_count: int
    """当前大轮已重试次数。"""


class BranchOutput(TypedDict, total=False):
    """每个子节点的输出。"""

    handler: Round1Handler
    """由哪个 handler 产生。"""

    reply: str
    """面向用户的回复文本（若有）。"""

    is_relevant: bool
    """AiThinking 判定：是否与身体状态相关。"""

    extracted_symptoms: list[str]
    """从用户话术中提取的症状词。"""

    body_part: BodyPart | None
    """推断或选定的部位。"""

    disease_type: str | None
    """推断的疾病类型。"""

    matched: bool
    """是否命中部位+疾病（match_agent 产出）。"""

    need_user_reply: bool
    """是否需要等待用户再次输入。"""


# ---------------------------------------------------------------------------
# 路由决策（类比 Classification）
# ---------------------------------------------------------------------------


class BranchDecision(TypedDict):
    """单次路由决策：走哪条路、为什么。"""

    route: Round1Route
    """一级：remind | continue。"""

    sub_route: ContinueSubRoute | None
    """二级：仅 continue 时有值。"""

    handler: Round1Handler
    """实际执行的 handler。"""

    reason: str
    """判定理由（可写入日志 / debug）。"""

    confidence: float
    """置信度 0~1。"""


# ---------------------------------------------------------------------------
# 主路由状态（类比 RouterState）
# ---------------------------------------------------------------------------


class Round1RouterState(TypedDict, total=False):
    """第一大轮 · 第一轮 · AiThinking 路由状态。

    由 relevance_checker 写入 decisions，
    各子 agent 写入 branch_results（自动累积）。
    """

    query: str
    """当前轮用户输入。"""

    input_mode: UserInputMode
    """当前输入方式。"""

    selected_body_part: BodyPart | None
    """UI 选项带来的部位（若有）。"""

    decisions: list[BranchDecision]
    """路由决策链（按时间追加）。"""

    branch_results: Annotated[list[BranchOutput], operator.add]
    """各子节点输出，自动累积。"""

    active_route: Round1Route | None
    """当前激活的一级路由。"""

    active_handler: Round1Handler | None
    """当前正在执行的 handler。"""


def initial_round1_router_state(
    query: str = "",
    input_mode: UserInputMode = "option",
) -> Round1RouterState:
    """初始化第一大轮第一轮路由状态。"""
    return Round1RouterState(
        query=query,
        input_mode=input_mode,
        selected_body_part=None,
        decisions=[],
        branch_results=[],
        active_route=None,
        active_handler=None,
    )
