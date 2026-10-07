"""8.5 注册入口：开业前把所有 ToolEntry 挂上注册表。"""

from __future__ import annotations

from typing import Any

from ..agent.sub_agents.disease_retrieval import create_disease_retrieval_agent
from ..agent.sub_agents.symptom_match import create_symptom_match_agent
from .medical_utils import (
    DESC_LIST_DISEASES_FOR_BODY_PART,
    DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG,
    DESC_SEARCH_DISEASES_BY_SYMPTOM,
    ListDiseasesForBodyPartArgs,
    LookupDiseaseSymptomCatalogArgs,
    SearchDiseasesBySymptomArgs,
    get_disease_detail,
    list_diseases_by_body_part,
    search_diseases_rag,
)
from .registry import ToolRegistry, get_registry
from .tool_entry import ToolMeta
from .tool_spec import ToolSpec

__all__ = ["register_all", "register_all_agents"]

# ---------------------------------------------------------------------------
# 子 Agent 内部 executor 的 ToolSpec（轻量查询 / RAG）
# ---------------------------------------------------------------------------

SEARCH_DISEASES_BY_SYMPTOM_SPEC = ToolSpec(
    name="search_diseases_by_symptom",
    description=DESC_SEARCH_DISEASES_BY_SYMPTOM,
    parameters=SearchDiseasesBySymptomArgs,
    visible_to="subagent:disease_retrieval",
)

LIST_DISEASES_FOR_BODY_PART_SPEC = ToolSpec(
    name="list_diseases_for_body_part",
    description=DESC_LIST_DISEASES_FOR_BODY_PART,
    parameters=ListDiseasesForBodyPartArgs,
    visible_to="subagent:disease_retrieval",
)

LOOKUP_DISEASE_SYMPTOM_CATALOG_SPEC = ToolSpec(
    name="lookup_disease_symptom_catalog",
    description=DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG,
    parameters=LookupDiseaseSymptomCatalogArgs,
    visible_to="subagent:symptom_match",
)


def search_diseases_by_symptom_executor(args: dict[str, Any]) -> Any:
    return search_diseases_rag.invoke(args)


def list_diseases_for_body_part_executor(args: dict[str, Any]) -> Any:
    return list_diseases_by_body_part.invoke(args)


def lookup_disease_symptom_catalog_executor(args: dict[str, Any]) -> Any:
    return get_disease_detail.invoke(args)


def register_all_agents(registry: ToolRegistry | None = None) -> ToolRegistry:
    """8.5 开业注册：子 Agent + executor 工具全部挂载到 ToolRegistry。"""
    if registry is None:
        registry = get_registry()

    # 子 Agent（主 Agent 委派入口）
    disease_retrieval = create_disease_retrieval_agent()
    registry.register_subagent(
        disease_retrieval,
        disease_retrieval.tool_spec,
        meta=disease_retrieval.tool_meta,
    )

    symptom_match = create_symptom_match_agent()
    registry.register_subagent(
        symptom_match,
        symptom_match.tool_spec,
        meta=symptom_match.tool_meta,
    )

    # 子 Agent 内部 executor 工具
    _register_medical_executors(registry)

    return registry


def register_all(registry: ToolRegistry) -> None:
    """兼容旧签名：挂载到已有 registry（由 ensure_tools_registered 调用）。"""
    register_all_agents(registry)


def _register_medical_executors(registry: ToolRegistry) -> None:
    registry.register_function(
        SEARCH_DISEASES_BY_SYMPTOM_SPEC,
        search_diseases_by_symptom_executor,
        meta=ToolMeta(
            title="按症状检索疾病",
            summary="已锁定部位后，用用户口语症状做 Milvus 语义检索，返回候选疾病。",
            phases=(1,),
            category="rag",
            operation_hint="第一大轮、已选部位、用户描述症状时优先调用。",
        ),
        langchain_tool=search_diseases_rag,
    )
    registry.register_function(
        LIST_DISEASES_FOR_BODY_PART_SPEC,
        list_diseases_for_body_part_executor,
        meta=ToolMeta(
            title="列出部位常见疾病",
            summary="按身体部位返回结构化疾病列表，不走向量检索。",
            phases=(1,),
            category="lookup",
            operation_hint="已选部位、需要展示疾病选项且用户未给足够症状描述时调用。",
        ),
        langchain_tool=list_diseases_by_body_part,
    )
    registry.register_function(
        LOOKUP_DISEASE_SYMPTOM_CATALOG_SPEC,
        lookup_disease_symptom_catalog_executor,
        meta=ToolMeta(
            title="查询疾病症状表",
            summary="按已锁定疾病名查询官方症状与同义词，用于症状匹配。",
            phases=(2,),
            category="lookup",
            operation_hint="第二大轮必须先调用，再判断 matched / candidate_symptoms。",
        ),
        langchain_tool=get_disease_detail,
    )
