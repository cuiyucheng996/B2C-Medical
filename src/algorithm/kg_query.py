"""公共知识：Neo4j / RAGFlow / Chroma 分源查询，结果给 LLM，不让模型编清单。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root
from algorithm.ragflow_kb import retrieve_chunks

logger = logging.getLogger(__name__)

_AGENT_HANDBOOK_ITEMS = 2
_AGENT_HANDBOOK_ITEM_CHARS = 300

__all__ = [
    "fulfill_kg_intent",
    "query_consult_graph",
    "search_exam_handbook",
    "search_chroma_knowledge",
    "slim_handbook_pack_for_agent",
]

_QUESTION_TAIL = (
    "常见有哪些症状",
    "有哪些常见疾病",
    "有哪些症状",
    "会有什么表现",
    "典型症状",
    "出现在哪个部位",
    "是哪个地方的疼",
    "通俗讲",
    "是什么病",
    "解释一下这个病",
    "检查什么",
    "是看什么",
    "是查什么",
    "正常值",
    "参考范围",
    "偏高是什么",
    "偏低是什么",
    "注意什么",
    "能不能运动",
    "饮食注意",
    "忌口",
    "证型表现",
    "什么证",
    "这个检查",
    "这个指标",
    "这个部位有哪些病",
    "常见有哪些病",
    "有哪些病",
    "会得什么病",
    "常见疾病",
    "请问",
    "帮我查",
    "查一下",
)


_GRAPH_HANDLERS = frozenset(
    {
        "kg_part_diseases",
        "kg_disease_symptoms",
        "kg_symptom_parts",
        "kg_disease_plain",
        "kg_syndrome_signs",
    }
)


def fulfill_kg_intent(
    *,
    handler: str,
    query: str,
    mentions: dict[str, list[str]],
    alignments: list[dict[str, Any]],
) -> dict[str, Any]:
    """兼容旧入口：图类走 Neo4j，其余走手册。新代码请调三个分源函数。"""
    name = (handler or "").strip() or "fallback"
    if name in _GRAPH_HANDLERS:
        return query_consult_graph(
            handler=name, query=query, mentions=mentions, alignments=alignments
        )
    return search_exam_handbook(query=query, mentions=mentions, alignments=alignments, handler=name)


def query_consult_graph(
    *,
    handler: str,
    query: str,
    mentions: dict[str, list[str]],
    alignments: list[dict[str, Any]],
) -> dict[str, Any]:
    """只查 Neo4j 问诊图。入口来自进工具前的抽实体与对齐，这里只跑 Cypher。"""
    name = (handler or "").strip() or "fallback"
    used_mentions = dict(mentions or {})
    used_alignments = list(alignments or [])
    seeds = _resolve_seeds(name, query, used_mentions, used_alignments)
    try:
        ensure_uvgraph_root()
        from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver
        from src.configs import NEO4J_CONFIG

        driver = get_neo4j_driver()
        db = str(NEO4J_CONFIG.get("database") or "neo4j")
    except Exception as exc:
        logger.exception("kg neo4j driver failed")
        return _pack(name, seeds, [], f"图谱不可用：{exc}", "neo4j", alignments=used_alignments)
    items: list[str] = []
    try:
        if name == "kg_part_diseases":
            items = _names_for_anchor(
                driver,
                db,
                anchor_label="ConsultBodyPart",
                anchor_prop="part_name",
                target_label="ConsultDisease",
                target_prop="disease_name",
                seeds=seeds,
            )
        elif name == "kg_disease_symptoms":
            items = _disease_symptoms(driver, db, seeds)
        elif name == "kg_symptom_parts":
            items = _symptom_parts(driver, db, seeds)
        elif name == "kg_disease_plain":
            items = _disease_plain(driver, db, seeds)
        elif name == "kg_syndrome_signs":
            items = _syndrome_signs(driver, db, seeds or _guess_seed(query))
        else:
            items, _source = _graph_scan(driver, db, query, seeds)
    except Exception as exc:
        logger.exception("kg graph query failed handler=%s", name)
        return _pack(name, seeds, [], f"查询失败：{exc}", "neo4j", alignments=used_alignments)
    pack = _pack(name, seeds, items, _summarize(name, seeds, items), "neo4j", alignments=used_alignments)
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import enhance_graph_pack

        if is_rag_enhanced():
            return enhance_graph_pack(pack, query=query, handler=name)
    except Exception:
        logger.warning("rag graph enhance failed", exc_info=True)
    return pack


def search_exam_handbook(
    *,
    query: str,
    mentions: dict[str, list[str]] | None = None,
    alignments: list[dict[str, Any]] | None = None,
    handler: str = "",
) -> dict[str, Any]:
    """只查 RAGFlow / 本地检查手册分块。"""
    name = (handler or "").strip() or "kg_exam_meaning"
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import run_rag_pipeline

        if is_rag_enhanced():
            return run_rag_pipeline(
                "handbook",
                query=query,
                handler=name,
                mentions=mentions,
                alignments=alignments,
            )
    except Exception:
        logger.warning("rag handbook pipeline failed", exc_info=True)
    seeds = _resolve_seeds(name, query, mentions or {}, alignments or [])
    needles = list(dict.fromkeys([*seeds, (query or "").strip()]))
    items = retrieve_chunks(" ".join(item for item in needles if item)[:200])
    return _pack(name, seeds, items, _summarize(name, seeds, items), "ragflow", alignments=alignments or [])


def search_chroma_knowledge(
    *,
    query: str,
    handler: str = "",
) -> dict[str, Any]:
    """只查 Chroma 疾病/症状向量。"""
    name = (handler or "").strip() or "fallback"
    text = (query or "").strip()
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import run_rag_pipeline

        if is_rag_enhanced():
            return run_rag_pipeline("chroma", query=text, handler=name)
    except Exception:
        logger.warning("rag chroma pipeline failed", exc_info=True)
    items = _chroma_snips(text)
    seeds = [text] if len(text) >= 2 else []
    return _pack(name, seeds, items, _summarize(name, seeds, items), "chroma")


def _pack(
    handler: str,
    seeds: list[str],
    items: list[str],
    summary: str,
    source: str,
    *,
    alignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "handler": handler,
        "source": source,
        "seeds": seeds,
        "items": items[:24],
        "summary": summary,
        "alignments": [
            {
                "mention": item.get("mention"),
                "standard_name": item.get("standard_name"),
                "node_label": item.get("node_label"),
                "method": item.get("method"),
            }
            for item in (alignments or [])
            if isinstance(item, dict)
        ],
    }


def slim_handbook_pack_for_agent(pack: dict[str, Any]) -> dict[str, Any]:
    """工具回执进 Agent 前裁剪：summary + 至多 2 条短摘录，去掉 rag/alignments/seeds 等。"""
    if not isinstance(pack, dict):
        return {"summary": "", "items": []}
    summary = str(pack.get("summary") or "").strip()
    if len(summary) > 400:
        summary = summary[:399].rstrip() + "…"
    items: list[str] = []
    for row in list(pack.get("items") or []):
        text = str(row if isinstance(row, str) else (row.get("text") if isinstance(row, dict) else row) or "").strip()
        if not text:
            continue
        if len(text) > _AGENT_HANDBOOK_ITEM_CHARS:
            text = text[: _AGENT_HANDBOOK_ITEM_CHARS - 1].rstrip() + "…"
        items.append(text)
        if len(items) >= _AGENT_HANDBOOK_ITEMS:
            break
    out: dict[str, Any] = {"summary": summary, "items": items}
    handler = str(pack.get("handler") or "").strip()
    source = str(pack.get("source") or "").strip()
    if handler:
        out["handler"] = handler[:64]
    if source:
        out["source"] = source[:32]
    return out


def _resolve_seeds(
    handler: str,
    query: str,
    mentions: dict[str, list[str]],
    alignments: list[dict[str, Any]],
) -> list[str]:
    wanted = {
        "kg_part_diseases": ("ConsultBodyPart", "部位"),
        "kg_disease_symptoms": ("ConsultDisease", "疾病"),
        "kg_disease_plain": ("ConsultDisease", "疾病"),
        "kg_symptom_parts": ("ConsultPrimarySymptom", "症状"),
        "kg_symptom_care": ("ConsultPrimarySymptom", "症状"),
        "kg_syndrome_signs": ("SyndromeType", ""),
        "kg_exam_meaning": ("", ""),
        "kg_lab_range": ("", ""),
    }.get(handler, ("", ""))
    label, mention_key = wanted
    names: list[str] = []
    for hit in alignments or []:
        if label and label not in str(hit.get("node_label") or ""):
            continue
        name = str(hit.get("standard_name") or hit.get("mention") or "").strip()
        if name:
            names.append(name)
    if mention_key:
        names.extend(str(item).strip() for item in (mentions or {}).get(mention_key) or [])
    guessed = _guess_seed(query)
    names.extend(guessed)
    return list(dict.fromkeys(item for item in names if item))[:6]


def _guess_seed(query: str) -> list[str]:
    text = (query or "").strip()
    for tail in sorted(_QUESTION_TAIL, key=len, reverse=True):
        text = text.replace(tail, "")
    text = re.sub(r"[?？。，,.、！!]+", "", text).strip()
    text = re.sub(r"^(这个|那个|该|请问)", "", text).strip()
    return [text] if len(text) >= 2 else []


def _names_for_anchor(
    driver: Any,
    db: str,
    *,
    anchor_label: str,
    anchor_prop: str,
    target_label: str,
    target_prop: str,
    seeds: list[str],
) -> list[str]:
    found: list[str] = []
    for seed in seeds:
        rows = _query(
            driver,
            db,
            f"""
            MATCH (a:{anchor_label})-[r:BELONG|Belong]-(b:{target_label})
            WHERE a.{anchor_prop} = $seed
               OR a.{anchor_prop} CONTAINS $seed
               OR $seed CONTAINS a.{anchor_prop}
            RETURN DISTINCT b.{target_prop} AS name
            LIMIT 24
            """,
            {"seed": seed},
        )
        found.extend(str(row.get("name") or "").strip() for row in rows)
    return list(dict.fromkeys(item for item in found if item))


def _disease_symptoms(driver: Any, db: str, seeds: list[str]) -> list[str]:
    names = _names_for_anchor(
        driver,
        db,
        anchor_label="ConsultDisease",
        anchor_prop="disease_name",
        target_label="ConsultPrimarySymptom",
        target_prop="symptom_name",
        seeds=seeds,
    )
    for seed in seeds:
        rows = _query(
            driver,
            db,
            """
            MATCH (st:SyndromeType)-[:BELONG|Belong]-(d:ConsultDisease)
            WHERE d.disease_name = $seed OR d.disease_name CONTAINS $seed
            RETURN st.main_symptoms AS main_symptoms
            LIMIT 12
            """,
            {"seed": seed},
        )
        for row in rows:
            names.extend(_parse_list(row.get("main_symptoms")))
    return list(dict.fromkeys(item for item in names if item))


def _symptom_parts(driver: Any, db: str, seeds: list[str]) -> list[str]:
    found: list[str] = []
    for seed in seeds:
        rows = _query(
            driver,
            db,
            """
            MATCH (s:ConsultPrimarySymptom)-[:BELONG|Belong]-(d:ConsultDisease)-[:BELONG|Belong]-(p:ConsultBodyPart)
            WHERE s.symptom_name = $seed
               OR s.symptom_name CONTAINS $seed
               OR $seed CONTAINS s.symptom_name
            RETURN DISTINCT p.part_name AS name
            LIMIT 16
            """,
            {"seed": seed},
        )
        found.extend(str(row.get("name") or "").strip() for row in rows)
    return list(dict.fromkeys(item for item in found if item))


def _disease_plain(driver: Any, db: str, seeds: list[str]) -> list[str]:
    lines: list[str] = []
    plain_map: dict[str, str] = {}
    try:
        from b2c_b2b_bridge import b2b_disease_plain

        plain_map = dict(b2b_disease_plain().DISEASE_PLAIN_EXPLAIN)
    except Exception:
        logger.debug("disease_plain table unavailable", exc_info=True)
    for seed in seeds:
        if seed in plain_map:
            lines.append(f"{seed}：{plain_map[seed]}")
        rows = _query(
            driver,
            db,
            """
            MATCH (d:ConsultDisease)
            WHERE d.disease_name = $seed OR d.disease_name CONTAINS $seed
            RETURN d.disease_name AS name, coalesce(d.description, '') AS description
            LIMIT 3
            """,
            {"seed": seed},
        )
        for row in rows:
            name = str(row.get("name") or seed).strip()
            desc = str(row.get("description") or "").strip()
            if desc:
                lines.append(f"{name}：{desc[:80]}")
            elif name and not any(name in item for item in lines):
                lines.append(name)
    return list(dict.fromkeys(lines))


def _syndrome_signs(driver: Any, db: str, seeds: list[str]) -> list[str]:
    lines: list[str] = []
    for seed in seeds:
        rows = _query(
            driver,
            db,
            """
            MATCH (st:SyndromeType)
            WHERE st.syndrome_name = $seed
               OR st.syndrome_name CONTAINS $seed
               OR $seed CONTAINS st.syndrome_name
            RETURN st.syndrome_name AS name,
                   st.main_symptoms AS main_symptoms,
                   st.supplementary_symptoms AS supplementary_symptoms
            LIMIT 8
            """,
            {"seed": seed},
        )
        for row in rows:
            name = str(row.get("name") or "").strip()
            mains = _parse_list(row.get("main_symptoms"))
            extras = _parse_list(row.get("supplementary_symptoms"))
            signs = list(dict.fromkeys([*mains, *extras]))
            if name and signs:
                lines.append(f"{name}：{'、'.join(signs[:12])}")
            elif name:
                lines.append(name)
    return lines


def _graph_scan(
    driver: Any,
    db: str,
    query: str,
    seeds: list[str],
) -> tuple[list[str], str]:
    extra = [query.strip()] if (query or "").strip() else []
    needles = list(dict.fromkeys([*seeds, *extra]))
    found: list[str] = []
    for seed in needles[:4]:
        if len(seed) < 2:
            continue
        rows = _query(
            driver,
            db,
            """
            MATCH (n)
            WHERE any(l IN labels(n) WHERE l STARTS WITH 'Consult'
               OR l STARTS WITH 'Syndrome'
               OR l STARTS WITH 'Report'
               OR l IN ['Disease', 'Symptom', 'BodyPart', 'Syndrome'])
              AND any(k IN keys(n) WHERE n[k] IS NOT NULL
                AND toString(n[k]) CONTAINS $seed)
            RETURN labels(n) AS labels, properties(n) AS props
            LIMIT 6
            """,
            {"seed": seed[:40]},
        )
        for row in rows:
            labels = [str(item) for item in (row.get("labels") or [])]
            props = row.get("props") if isinstance(row.get("props"), dict) else {}
            snippet = _props_snippet(labels, props, seed)
            if snippet:
                found.append(snippet)
    return list(dict.fromkeys(found))[:10], "neo4j"


def _chroma_snips(query: str) -> list[str]:
    from algorithm.chroma_knowledge import chroma_knowledge_snippets

    return chroma_knowledge_snippets(query, limit=4)


def _props_snippet(labels: list[str], props: dict[str, Any], seed: str) -> str:
    skip = {"id", "status", "created_at", "updated_at", "sort_order"}
    bits: list[str] = []
    for key, value in props.items():
        if key in skip or value is None:
            continue
        text = str(value).strip()
        if not text or (len(text) > 120 and seed not in text):
            continue
        if seed in text or key.endswith("name") or key in {"description", "meaning", "range", "reference"}:
            bits.append(f"{key}={text[:80]}")
        if len(bits) >= 4:
            break
    if not bits:
        return ""
    label = "/".join(labels[:2]) or "Node"
    return f"{label} " + "；".join(bits)


def _summarize(handler: str, seeds: list[str], items: list[str]) -> str:
    topic = "、".join(seeds[:3]) or "该问句"
    if not items:
        return f"未查到「{topic}」的条目，回答时不要编造清单。"
    joined = "、".join(items[:16])
    titles = {
        "kg_part_diseases": f"部位「{topic}」图谱中的常见疾病",
        "kg_disease_symptoms": f"疾病「{topic}」图谱中的主要症状",
        "kg_symptom_parts": f"症状「{topic}」图谱中的常见部位",
        "kg_disease_plain": f"疾病「{topic}」的通俗说明",
        "kg_syndrome_signs": f"证型「{topic}」图谱中的常见表现",
        "kg_exam_meaning": f"与检查「{topic}」相关的知识命中",
        "kg_lab_range": f"与指标「{topic}」相关的知识命中",
        "kg_symptom_care": f"与症状「{topic}」相关的知识命中",
    }
    head = titles.get(handler, f"「{topic}」知识命中")
    return f"{head}：{joined}"


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in re.split(r"[、,，|]", text) if part.strip()]


def _query(driver: Any, db: str, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    result = driver.execute_query(cypher, parameters_=params, database_=db)
    return [row.data() for row in result.records]
