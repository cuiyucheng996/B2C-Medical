"""进场长期记忆收集：按实体计数，并用图谱主症/辅症重叠推断证型。

潜在证型按「同一部位下的疾病」合计：该部位候选多于 3 个时，这一部位不写「可能性最大」。逻辑独立于 B 端问诊轮次。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_WORKER_MEMORY_SUMMARY_CHARS = 300

MEMORY_COLLECT_PREFIX = "【长期记忆的收集】"

# 仅子 Agent 按需注入长期记忆摘要（不从会话 history 复用进场大树）
_WORKER_MEMORY_HANDLERS = frozenset({"status_complaint", "mem_session_recall"})

__all__ = [
    "MEMORY_COLLECT_PREFIX",
    "compose_memory_collect",
    "history_for_intent_worker",
    "rank_syndrome_candidates",
    "shrink_memory_collect_text",
    "strip_memory_collect_messages",
]

_MAX_CANDIDATES = 3
_USER_MATCH = "toString(u.id) = $user_id OR coalesce(u.login_name, '') = $user_id"
_IND1 = "          "
_IND2 = "                   "
_IND3 = "                            "


def compose_memory_collect(user_id: str) -> str:
    uid = (user_id or "").strip()
    if not uid:
        return ""
    buckets, maps = _count_entities(uid)
    body = _format_tree(buckets, maps)
    if not body:
        return (
            "【长期记忆的收集】\n"
            "还没有可计数的部位、疾病或症状。你可以直接说说近况，结束会话后我会记下来。\n"
            "这不是诊断。"
        )
    return f"【长期记忆的收集】\n{body}\n这不是诊断，只是根据已记下的实体和图谱条目做的归纳。"


def shrink_memory_collect_text(text: str, max_chars: int = _WORKER_MEMORY_SUMMARY_CHARS) -> str:
    """进场长期记忆树压成短摘要，供子 Agent 上下文。"""
    raw = (text or "").strip()
    if not raw.startswith(MEMORY_COLLECT_PREFIX):
        return raw[:max_chars] if max_chars > 0 and len(raw) > max_chars else raw
    limit = max(80, int(max_chars))
    body = raw[len(MEMORY_COLLECT_PREFIX) :].strip()
    tail = "这不是诊断，只是根据已记下的实体和图谱条目做的归纳。"
    if body.endswith(tail):
        body = body[: -len(tail)].strip()
    one_line = " ".join(line.strip() for line in body.splitlines() if line.strip())
    if len(one_line) <= limit:
        short_body = one_line
    else:
        short_body = one_line[: limit - 1].rstrip() + "…"
    return f"{MEMORY_COLLECT_PREFIX}\n{short_body}\n（摘要已截断，非诊断。）"


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def strip_memory_collect_messages(history: list[Any]) -> list[Any]:
    """意图、语义缓存、主图 history 等均不应带进场长期记忆树。"""
    out: list[Any] = []
    for msg in history or []:
        if _message_text(msg).startswith(MEMORY_COLLECT_PREFIX):
            continue
        out.append(msg)
    return out


def history_for_intent_worker(
    history: list[Any],
    *,
    handler: str = "",
    major: str = "",
    user_id: str = "",
) -> list[Any]:
    """子 Agent：剥离会话中的进场树；仅特定 handler 用 compose 注入短摘要。"""
    del major
    from langchain_core.messages import AIMessage

    out = strip_memory_collect_messages(history)
    h = (handler or "").strip()
    uid = (user_id or "").strip()
    if h not in _WORKER_MEMORY_HANDLERS or not uid:
        return out
    block = compose_memory_collect(uid)
    if not block.strip():
        return out
    short = shrink_memory_collect_text(block, _WORKER_MEMORY_SUMMARY_CHARS)
    if short:
        out.insert(0, AIMessage(content=short))
    return out


def rank_syndrome_candidates(
    symptom_weights: dict[str, int],
    syndromes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按用户症状权重与证型主症/辅症的交集打分，高分在前。"""
    weights = {name: int(times) for name, times in symptom_weights.items() if name and int(times) > 0}
    ranked: list[tuple[int, dict[str, Any]]] = []
    for row in syndromes:
        name = str(row.get("syndrome_name") or "").strip()
        if not name:
            continue
        mains = _as_list(row.get("main_symptoms"))
        extras = _as_list(row.get("supplementary_symptoms"))
        score = 0
        hit_main: list[str] = []
        hit_aux: list[str] = []
        for symptom, times in weights.items():
            if symptom in mains:
                score += times * 2
                hit_main.append(symptom)
            elif symptom in extras:
                score += times
                hit_aux.append(symptom)
        if score <= 0:
            continue
        ranked.append(
            (
                score,
                {
                    "syndrome_name": name,
                    "score": score,
                    "main_symptoms": mains,
                    "supplementary_symptoms": extras,
                    "hit_main": hit_main,
                    "hit_aux": hit_aux,
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]["syndrome_name"]))
    return [item for _, item in ranked]


def _format_tree(
    buckets: dict[str, list[tuple[str, int]]],
    maps: dict[str, Any],
) -> str:
    tree = _build_memory_tree(buckets, maps)
    if not tree:
        return ""
    syndromes = _load_syndromes()
    weights = _symptom_weights(buckets)
    lines: list[str] = []
    for part, block in tree:
        pc = int(block.get("count") or 0)
        lines.append(f"{part}：{pc}次" if pc > 0 else f"{part}：")
        diseases = list(block.get("diseases") or [])
        for disease, row in diseases:
            dc = int(row.get("count") or 0)
            lines.append(f"{_IND1}{disease}：{dc}次" if dc > 0 else f"{_IND1}{disease}：")
            mains = row.get("main") or []
            auxs = row.get("aux") or []
            if mains or auxs:
                lines.append(f"{_IND2}主症：")
                for sym, times in mains:
                    lines.append(f"{_IND3}{sym}：{times}次")
                lines.append(f"{_IND2}辅症：")
                for sym, times in auxs:
                    lines.append(f"{_IND3}{sym}：{times}次")
        hint = _syndrome_hint_for_diseases([name for name, _ in diseases], weights, syndromes)
        if hint:
            lines.append(f"{_IND1}{hint}")
    return "\n".join(lines)


def _build_memory_tree(
    buckets: dict[str, list[tuple[str, int]]],
    maps: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    part_counts = dict(buckets.get("body_part") or [])
    disease_counts = dict(buckets.get("disease") or [])
    main_counts = dict(buckets.get("primary_symptom") or [])
    aux_counts = dict(buckets.get("auxiliary_symptom") or [])
    if not part_counts and not disease_counts:
        return []

    part_to_diseases: dict[str, set[str]] = maps.get("part_to_diseases") or {}
    disease_to_parts: dict[str, set[str]] = maps.get("disease_to_parts") or {}
    disease_to_mains: dict[str, set[str]] = maps.get("disease_to_mains") or {}
    disease_to_aux: dict[str, set[str]] = maps.get("disease_to_aux") or {}
    symptom_to_diseases: dict[str, set[str]] = maps.get("symptom_to_diseases") or {}
    unlinked = "（图谱未挂部位）"

    part_order = [name for name, _ in buckets.get("body_part") or [] if name in part_counts]

    nodes: dict[str, dict[str, Any]] = {}
    for part in part_order:
        nodes[part] = {"count": part_counts.get(part, 0), "diseases": {}}

    def _ensure_part(part: str) -> dict[str, Any]:
        if part not in nodes:
            nodes[part] = {"count": part_counts.get(part, 0), "diseases": {}}
            part_order.append(part)
        return nodes[part]

    def _ensure_disease(part: str, disease: str, count: int) -> dict[str, Any] | None:
        if disease not in disease_counts:
            return None
        if part != unlinked:
            allowed = part_to_diseases.get(part) or set()
            if allowed and disease not in allowed:
                return None
        block = _ensure_part(part)
        diseases = block["diseases"]
        if disease not in diseases:
            diseases[disease] = {"count": 0, "main": {}, "aux": {}}
        diseases[disease]["count"] = max(int(diseases[disease]["count"]), count)
        return diseases[disease]

    for disease, dcount in disease_counts.items():
        parts = sorted(disease_to_parts.get(disease) or [])
        if not parts:
            _ensure_disease(unlinked, disease, dcount)
            continue
        for part in parts:
            _ensure_disease(part, disease, dcount)

    for sym, scount in main_counts.items():
        candidates = symptom_to_diseases.get(sym) or _diseases_for_main(sym, disease_to_mains)
        for disease in sorted(candidates):
            if disease not in disease_counts:
                continue
            for part, block in nodes.items():
                row = (block.get("diseases") or {}).get(disease)
                if row is None:
                    continue
                row["main"][sym] = row["main"].get(sym, 0) + scount

    for sym, scount in aux_counts.items():
        for disease, aux_set in disease_to_aux.items():
            if disease not in disease_counts or sym not in aux_set:
                continue
            for block in nodes.values():
                row = (block.get("diseases") or {}).get(disease)
                if row is None:
                    continue
                row["aux"][sym] = row["aux"].get(sym, 0) + scount

    out: list[tuple[str, dict[str, Any]]] = []
    for part in part_order:
        block = nodes.get(part)
        if not block:
            continue
        disease_items: list[tuple[str, dict[str, Any]]] = []
        for name, row in (block.get("diseases") or {}).items():
            if name not in disease_counts:
                continue
            main_list = sorted(
                ((k, v) for k, v in (row.get("main") or {}).items() if v > 0),
                key=lambda item: (-item[1], item[0]),
            )
            aux_list = sorted(
                ((k, v) for k, v in (row.get("aux") or {}).items() if v > 0),
                key=lambda item: (-item[1], item[0]),
            )
            disease_items.append(
                (
                    name,
                    {
                        "count": int(row.get("count") or 0),
                        "main": main_list,
                        "aux": aux_list,
                    },
                )
            )
        disease_items.sort(key=lambda item: (-item[1]["count"], item[0]))
        out.append((part, {"count": int(block.get("count") or 0), "diseases": disease_items}))
    return out


def _diseases_for_main(symptom: str, disease_to_mains: dict[str, set[str]]) -> set[str]:
    hits: set[str] = set()
    for disease, names in disease_to_mains.items():
        if symptom in names:
            hits.add(disease)
    return hits


def _load_consult_maps(driver: Any, db: str) -> dict[str, Any]:
    part_to_diseases: dict[str, set[str]] = {}
    disease_to_parts: dict[str, set[str]] = {}
    disease_to_mains: dict[str, set[str]] = {}
    disease_to_aux: dict[str, set[str]] = {}
    symptom_to_diseases: dict[str, set[str]] = {}

    try:
        rows = driver.execute_query(
            """
            MATCH (p:ConsultBodyPart)-[:BELONG|Belong]-(d:ConsultDisease)
            WHERE coalesce(p.status, 1) = 1 AND coalesce(d.status, 1) = 1
            RETURN p.part_name AS part, d.disease_name AS disease
            """,
            database_=db,
        )
        for rec in rows.records:
            data = rec.data()
            part = str(data.get("part") or "").strip()
            disease = str(data.get("disease") or "").strip()
            if not part or not disease:
                continue
            part_to_diseases.setdefault(part, set()).add(disease)
            disease_to_parts.setdefault(disease, set()).add(part)

        rows = driver.execute_query(
            """
            MATCH (d:ConsultDisease)-[:BELONG|Belong]-(s:ConsultPrimarySymptom)
            WHERE coalesce(d.status, 1) = 1 AND coalesce(s.status, 1) = 1
            RETURN d.disease_name AS disease, s.symptom_name AS symptom
            """,
            database_=db,
        )
        for rec in rows.records:
            data = rec.data()
            disease = str(data.get("disease") or "").strip()
            symptom = str(data.get("symptom") or "").strip()
            if not disease or not symptom:
                continue
            disease_to_mains.setdefault(disease, set()).add(symptom)
            symptom_to_diseases.setdefault(symptom, set()).add(disease)

        rows = driver.execute_query(
            """
            MATCH (st:SyndromeType)-[:BELONG|Belong]-(d:ConsultDisease)
            WHERE coalesce(st.status, 1) = 1 AND coalesce(d.status, 1) = 1
            RETURN d.disease_name AS disease,
                   st.main_symptoms AS main_symptoms,
                   st.supplementary_symptoms AS supplementary_symptoms
            """,
            database_=db,
        )
        for rec in rows.records:
            data = rec.data()
            disease = str(data.get("disease") or "").strip()
            if not disease:
                continue
            for name in _as_list(data.get("main_symptoms")):
                disease_to_mains.setdefault(disease, set()).add(name)
                symptom_to_diseases.setdefault(name, set()).add(disease)
            for name in _as_list(data.get("supplementary_symptoms")):
                disease_to_aux.setdefault(disease, set()).add(name)
    except Exception:
        logger.exception("memory collect consult maps failed")

    return {
        "part_to_diseases": part_to_diseases,
        "disease_to_parts": disease_to_parts,
        "disease_to_mains": disease_to_mains,
        "disease_to_aux": disease_to_aux,
        "symptom_to_diseases": symptom_to_diseases,
    }


def _symptom_weights(buckets: dict[str, list[tuple[str, int]]]) -> dict[str, int]:
    weights: dict[str, int] = {}
    for key in ("primary_symptom", "auxiliary_symptom"):
        for name, times in buckets.get(key) or []:
            weights[name] = weights.get(name, 0) + times
    return weights


def _syndrome_hint_for_diseases(
    diseases: list[str],
    weights: dict[str, int],
    syndromes: list[dict[str, Any]],
) -> str:
    """同一部位下疾病的证型候选：超过 3 个则这一部位不展示。"""
    owned = {name for name in diseases if name}
    if not owned or not weights:
        return ""
    subset = [
        row
        for row in syndromes
        if str(row.get("disease_name") or "").strip() in owned
    ]
    ranked = rank_syndrome_candidates(weights, subset)
    if not ranked or len(ranked) > _MAX_CANDIDATES:
        return ""
    top = ranked[0]
    return f"您{top['syndrome_name']}证型可能性最大。{_trait_line(top)}"


def _trait_line(row: dict[str, Any]) -> str:
    mains = row.get("main_symptoms") or []
    extras = row.get("supplementary_symptoms") or []
    parts: list[str] = []
    if mains:
        parts.append("图谱中的主要表现：" + "、".join(str(item) for item in mains[:12]))
    if extras:
        parts.append("辅助表现：" + "、".join(str(item) for item in extras[:12]))
    if not parts:
        return "图谱里暂无该证型的表现清单。"
    return "".join(parts) + "。"


def _count_entities(user_id: str) -> tuple[dict[str, list[tuple[str, int]]], dict[str, Any]]:
    empty: dict[str, list[tuple[str, int]]] = {
        "disease": [],
        "body_part": [],
        "primary_symptom": [],
        "auxiliary_symptom": [],
    }
    maps: dict[str, Any] = {}
    try:
        driver, db = _driver_db()
    except Exception:
        logger.exception("memory collect driver failed")
        return empty, maps
    try:
        result = driver.execute_query(
            f"""
            MATCH (u:UserInfo)-[:HAVE]->(f:Finding)-[e:EXTRACTED]->(n)
            WHERE {_USER_MATCH}
            RETURN e.kind AS kind,
                   coalesce(n.part_name, n.disease_name, n.symptom_name, e.name) AS name,
                   count(DISTINCT f) AS times
            """,
            parameters_={"user_id": user_id},
            database_=db,
        )
    except Exception:
        logger.exception("memory collect count failed user_id=%s", user_id)
        return empty, maps

    primary_names: set[str] = set()
    for rec in result.records:
        data = rec.data()
        kind = str(data.get("kind") or "")
        name = str(data.get("name") or "").strip()
        times = int(data.get("times") or 0)
        if not name or times <= 0:
            continue
        if kind == "disease":
            empty["disease"].append((name, times))
        elif kind == "body_part":
            empty["body_part"].append((name, times))
        elif kind == "symptom":
            empty["primary_symptom"].append((name, times))
            primary_names.add(name)

    aux = _count_auxiliary(driver, db, user_id, primary_names)
    empty["auxiliary_symptom"] = aux
    for key in empty:
        empty[key].sort(key=lambda item: (-item[1], item[0]))
    maps = _load_consult_maps(driver, db)
    return empty, maps


def _count_auxiliary(driver: Any, db: str, user_id: str, primary_names: set[str]) -> list[tuple[str, int]]:
    catalog = _auxiliary_catalog(driver, db)
    if not catalog:
        return []
    try:
        result = driver.execute_query(
            f"""
            MATCH (u:UserInfo)-[:HAVE]->(f:Finding)
            WHERE {_USER_MATCH}
            RETURN f.raw_text AS raw_text
            """,
            parameters_={"user_id": user_id},
            database_=db,
        )
    except Exception:
        logger.exception("memory collect aux scan failed")
        return []
    counts: dict[str, int] = {}
    for rec in result.records:
        text = str(rec.data().get("raw_text") or "")
        if not text:
            continue
        for name in catalog:
            if name in primary_names or name not in text:
                continue
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _auxiliary_catalog(driver: Any, db: str) -> set[str]:
    names: set[str] = set()
    try:
        result = driver.execute_query(
            """
            MATCH (st:SyndromeType)
            WHERE coalesce(st.status, 1) = 1
            RETURN st.supplementary_symptoms AS supplementary_symptoms
            """,
            database_=db,
        )
    except Exception:
        logger.exception("memory collect aux catalog failed")
        return names
    for rec in result.records:
        for name in _as_list(rec.data().get("supplementary_symptoms")):
            if len(name) >= 2:
                names.add(name)
    return names


def _load_syndromes() -> list[dict[str, Any]]:
    try:
        driver, db = _driver_db()
        result = driver.execute_query(
            """
            MATCH (st:SyndromeType)
            WHERE coalesce(st.status, 1) = 1
            RETURN st.syndrome_name AS syndrome_name,
                   st.disease_name AS disease_name,
                   st.main_symptoms AS main_symptoms,
                   st.supplementary_symptoms AS supplementary_symptoms
            """,
            database_=db,
        )
    except Exception:
        logger.exception("memory collect syndromes failed")
        return []
    return [rec.data() for rec in result.records]


def _driver_db():
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.configs import NEO4J_CONFIG
    from src.RAGgraph.driver.neo4j_driver import get_neo4j_driver

    return get_neo4j_driver(), str(NEO4J_CONFIG.get("database") or "neo4j")


def _as_list(value: Any) -> list[str]:
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
