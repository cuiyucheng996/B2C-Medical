"""医疗问诊 LangChain Tools：Milvus RAG 检索 + 结构化疾病查询。

Milvus 实际检索在子进程执行（见 ``algorithm.milvus_subprocess``），主进程不 import pymilvus。

每个 Tool 均具备：
- ``name``：面向 LLM 的直观工具名（``@tool`` 第一参数）
- ``description``：何时调用、勿用场景、参数怎么填
- ``args_schema``：Pydantic JSON Schema（类型、required、枚举、字段说明）

Milvus collection 默认 ``disease``（与 ``ingest_mysql_milvus.py --table disease`` 一致）。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pymysql
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field
from pymysql.cursors import DictCursor

from src.configs import MYSQL_CONFIG, MYSQL_INGEST_DATABASE
from src.knowledge.body_disease_map import BODY_PARTS, diseases_for_body_part
from src.knowledge.md_mysql_ingest import quote_ident
from src.schemas import BodyPart

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5

_BODY_PART_HINT = "、".join(BODY_PARTS)

# Tool descriptions（LLM 选工具的主要依据，与 system prompt 对齐）
DESC_SEARCH_DISEASES_BY_SYMPTOM = """\
【何时使用】第一大轮 AiThinking：用户描述与身体不适/症状相关，且会话上下文**已锁定 body_part** 时，必须用本工具检索候选疾病。
【勿用】① 尚未选定部位（应自行推断部位，勿调工具）② 纯寒暄/感谢/无关话题 ③ 描述过短或过于模糊 ④ 敏感违规内容。
【参数怎么填】
- query：用户本轮症状或不适的**原文**（自然语言，如「脑袋胀痛、晚上睡不着」），不要自行改写疾病名。
- body_part：必填，必须与上下文「已选定部位」**完全一致**（八大部位枚举之一）。
- limit：可选，返回候选疾病条数，默认 5，范围 1～10。
【返回】Milvus 语义检索命中的疾病候选列表；无命中时返回提示文案。"""

_DESC_SEARCH_DISEASES_BY_SYMPTOM = DESC_SEARCH_DISEASES_BY_SYMPTOM

DESC_LIST_DISEASES_FOR_BODY_PART = """\
【何时使用】第一大轮：用户**已明确选定某一身体部位**，需要列出该部位下常见疾病选项供点选时；走结构化查询，**不做**向量语义检索。
【勿用】① 尚未选定部位 ② 已有口语症状描述、需要按症状猜疾病（应改用 search_diseases_by_symptom）③ 已锁定具体疾病名、需要查症状表（应改用 lookup_disease_symptom_catalog）。
【参数怎么填】
- body_part：必填，八大部位枚举之一，必须与用户已选部位完全一致。
【返回】该部位常见疾病名称列表；部位无效时返回可选部位提示。"""

_DESC_LIST_DISEASES_FOR_BODY_PART = DESC_LIST_DISEASES_FOR_BODY_PART

DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG = """\
【何时使用】第二大轮 AiThinking：用户**已锁定疾病类型**，需把口语症状映射到官方症状名时，**必须先调用本工具**获取该疾病的「包含症状」「通俗同义词」等字段，再判断 matched / candidate_symptoms。
【勿用】① 尚未锁定疾病 ② 仅需按部位列疾病（应改用 list_diseases_for_body_part）③ 按口语猜疾病（应改用 search_diseases_by_symptom）。
【参数怎么填】
- disease_name：必填，必须与上下文「已锁定疾病」**完全一致**（知识库标准病名，如「喘证」「头痛」）。
【返回】MySQL 疾病结构化详情（症状类型、包含症状、通俗同义词等）；未找到时返回提示文案。"""

_DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG = DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG


# ---------------------------------------------------------------------------
# JSON Schema（Pydantic args_schema）
# ---------------------------------------------------------------------------


class SearchDiseasesBySymptomArgs(BaseModel):
    """按口语症状在已锁定部位内做 Milvus 语义检索，返回候选疾病。"""

    model_config = ConfigDict(
        json_schema_extra={
            "description": "第一大轮：已选部位 + 用户症状描述 → 检索候选疾病（RAG）。",
        },
    )

    query: str = Field(
        min_length=1,
        max_length=500,
        description="用户本轮症状或不适描述原文（自然语言），如「最近头痛、晚上睡不着」。",
    )
    body_part: BodyPart = Field(
        description=f"已锁定的身体部位，必须与上下文一致。可选值：{_BODY_PART_HINT}。",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最多返回几条候选疾病，默认 5。",
    )


class ListDiseasesForBodyPartArgs(BaseModel):
    """按身体部位列出常见疾病（结构化，不走向量检索）。"""

    model_config = ConfigDict(
        json_schema_extra={
            "description": "第一大轮：已选部位 → 列出该部位常见疾病选项。",
        },
    )

    body_part: BodyPart = Field(
        description=f"用户已选身体部位。必须为以下之一：{_BODY_PART_HINT}。",
    )


class LookupDiseaseSymptomCatalogArgs(BaseModel):
    """按疾病名查询官方症状与同义词（第二大轮症状匹配前置步骤）。"""

    model_config = ConfigDict(
        json_schema_extra={
            "description": "第二大轮：已锁定疾病 → 查询官方症状表与同义词。",
        },
    )

    disease_name: str = Field(
        min_length=1,
        max_length=64,
        description="已锁定的标准疾病名称，须与知识库病名完全一致，如「喘证」「头痛」。",
    )


# 兼容旧 import 名
SearchDiseasesRagArgs = SearchDiseasesBySymptomArgs
ListDiseasesByBodyPartArgs = ListDiseasesForBodyPartArgs
GetDiseaseDetailArgs = LookupDiseaseSymptomCatalogArgs


# ---------------------------------------------------------------------------
# 实现（门面 / MySQL）
# ---------------------------------------------------------------------------


def search_diseases_in_milvus(
    query: str,
    *,
    body_part: str | None = None,
    limit: int = DEFAULT_TOP_K,
    max_distance: float = 0.65,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """Milvus 疾病 RAG 门面：在子进程中执行检索。"""
    from src.algorithm.milvus_subprocess import run_milvus_search_isolated

    return run_milvus_search_isolated(
        query,
        body_part=body_part,
        limit=limit,
        max_distance=max_distance,
        collection_name=collection_name,
    )


def _mysql_disease_database() -> str:
    return (MYSQL_INGEST_DATABASE or MYSQL_CONFIG.get("database") or "medical").strip()


def fetch_disease_from_mysql(disease_name: str) -> dict[str, Any] | None:
    """从 MySQL disease 表按疾病名精确查一行。"""
    name = disease_name.strip()
    if not name:
        return None

    db_name = _mysql_disease_database()
    sql = f"SELECT * FROM {quote_ident('disease')} WHERE {quote_ident('疾病')} = %s LIMIT 1"
    conn_config = {
        "host": MYSQL_CONFIG["host"],
        "user": MYSQL_CONFIG["user"],
        "password": MYSQL_CONFIG["password"],
        "charset": MYSQL_CONFIG["charset"],
        "port": MYSQL_CONFIG["port"],
        "database": db_name,
    }
    with pymysql.connect(**conn_config) as conn:
        with conn.cursor(DictCursor) as cursor:
            cursor.execute(sql, (name,))
            row = cursor.fetchone()
    if not row:
        return None
    return {
        str(k): float(v) if isinstance(v, Decimal) else v
        for k, v in row.items()
    }


# ---------------------------------------------------------------------------
# LangChain Tools（name / description / args_schema 三要素齐全）
# ---------------------------------------------------------------------------


@tool(
    "search_diseases_by_symptom",
    args_schema=SearchDiseasesBySymptomArgs,
    description=_DESC_SEARCH_DISEASES_BY_SYMPTOM,
)
def search_diseases_rag(
    query: str,
    body_part: BodyPart,
    limit: int = 5,
) -> list[dict[str, Any]] | str:
    hits = search_diseases_in_milvus(query, body_part=body_part, limit=limit)
    if hits:
        return hits
    return "未检索到与描述匹配的疾病候选，请补充更具体的症状或选择部位。"


@tool(
    "list_diseases_for_body_part",
    args_schema=ListDiseasesForBodyPartArgs,
    description=_DESC_LIST_DISEASES_FOR_BODY_PART,
)
def list_diseases_by_body_part(body_part: BodyPart) -> list[str] | str:
    part = str(body_part).strip()
    options = diseases_for_body_part(part)
    if options:
        return options
    valid = "、".join(BODY_PARTS)
    return f"未找到部位「{part}」的疾病列表。可选部位：{valid}"


@tool(
    "lookup_disease_symptom_catalog",
    args_schema=LookupDiseaseSymptomCatalogArgs,
    description=_DESC_LOOKUP_DISEASE_SYMPTOM_CATALOG,
)
def get_disease_detail(disease_name: str) -> dict[str, Any] | str:
    row = fetch_disease_from_mysql(disease_name)
    if row:
        return row
    return f"未在知识库中找到疾病「{disease_name.strip()}」的详情。"


MEDICAL_TOOLS = [
    search_diseases_rag,
    list_diseases_by_body_part,
    get_disease_detail,
]

__all__ = [
    "MEDICAL_TOOLS",
    "GetDiseaseDetailArgs",
    "ListDiseasesByBodyPartArgs",
    "ListDiseasesForBodyPartArgs",
    "LookupDiseaseSymptomCatalogArgs",
    "SearchDiseasesBySymptomArgs",
    "SearchDiseasesRagArgs",
    "fetch_disease_from_mysql",
    "get_disease_detail",
    "list_diseases_by_body_part",
    "search_diseases_in_milvus",
    "search_diseases_rag",
]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    sample_query = sys.argv[1] if len(sys.argv) > 1 else "最近脑袋胀痛，还偶尔头晕"
    print("=== search_diseases_by_symptom ===")
    print(search_diseases_rag.invoke({"query": sample_query, "body_part": "头面部", "limit": 3}))
    print("=== list_diseases_for_body_part ===")
    print(list_diseases_by_body_part.invoke({"body_part": "头面部"}))
    print("=== lookup_disease_symptom_catalog ===")
    print(get_disease_detail.invoke({"disease_name": "头痛"}))
