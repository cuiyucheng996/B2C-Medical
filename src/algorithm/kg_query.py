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
    hops = _one_hop_lines(driver, db, _aligned_names(used_alignments) or seeds)
    ranked = _rank_handler_overlap(driver, db, name, seeds, query)
    ranked.extend(_ranked_business_paths(driver, db, used_mentions))
    items = list(dict.fromkeys([*ranked, *items, *hops]))
    pack = _pack(name, seeds, items, _summarize(name, seeds, items), "neo4j", alignments=used_alignments)
    try:
        from algorithm.rag.config import is_rag_enhanced
        from algorithm.rag.pipeline import enhance_graph_pack

        if is_rag_enhanced():
            pack = enhance_graph_pack(pack, query=query, handler=name)
            front = [line for line in ranked if line in set(pack.get("items") or [])]
            rest = [line for line in list(pack.get("items") or []) if line not in set(front)]
            pack["items"] = (front + rest)[:24]
            return pack
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
    """实现在 server/builtin_server/report_read/handbook.py。"""
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from server.builtin_server.report_read.handbook import search_exam_handbook as run

    return run(query=query, mentions=mentions, alignments=alignments, handler=handler)


def search_chroma_knowledge(
    *,
    query: str,
    handler: str = "",
    alignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """实现在 server/builtin_server/report_read/handbook.py。"""
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from server.builtin_server.report_read.handbook import search_chroma_knowledge as run

    return run(query=query, handler=handler, alignments=alignments)


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


def _mention_list(mentions: dict[str, list[str]], key: str) -> list[str]:
    raw = mentions.get(key) if isinstance(mentions, dict) else None
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))[:6]


def _user_finding_blob(driver: Any, db: str) -> str:
    try:
        from tools.companion_memory import companion_user_id
    except Exception:
        return ""
    uid = str(companion_user_id.get() or "").strip()
    if not uid:
        return ""
    rows = _query(
        driver,
        db,
        """
        MATCH (u:UserInfo)-[:HAVE]->(f:Finding)
        WHERE toString(u.id) = $user_id
        RETURN f.raw_text AS raw_text
        ORDER BY f.created_at DESC
        LIMIT 8
        """,
        {"user_id": uid},
    )
    return " ".join(str(row.get("raw_text") or "") for row in rows)


def _char_jaccard(left: str, right: str) -> float:
    a = {ch for ch in left if not ch.isspace()}
    b = {ch for ch in right if not ch.isspace()}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(x * y for x, y in zip(left, right, strict=False))
    na = sum(x * x for x in left) ** 0.5
    nb = sum(y * y for y in right) ** 0.5
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _blend_path_scores(query: str, scored: list[tuple[float, str]]) -> list[str]:
    """票数为主；语义相似度拉开接近的票，字符 Jaccard 只做最后的同分。"""
    if not scored:
        return []
    ask = (query or "").strip()
    vectors: list[list[float]] = []
    try:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.embeddings import get_embeddings

        emb = get_embeddings()
        qv = emb.embed_query(ask) if ask else []
        docs = emb.embed_documents([line for _vote, line in scored])
        if qv and len(docs) == len(scored):
            vectors = [list(qv), *[list(item) for item in docs]]
    except Exception:
        logger.debug("graph path embed failed", exc_info=True)
        vectors = []
    blended: list[tuple[float, float, float, str]] = []
    qv = vectors[0] if vectors else []
    for idx, (vote, line) in enumerate(scored):
        semantic = _cosine(qv, vectors[idx + 1]) if qv and idx + 1 < len(vectors) else 0.0
        overlap = _char_jaccard(ask, line) if ask else 0.0
        blended.append((vote + semantic, overlap, vote, line))
    blended.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    out: list[str] = []
    for _mix, _overlap, _vote, line in blended:
        if line not in out:
            out.append(line)
        if len(out) >= 3:
            break
    return out


def _rank_handler_overlap(
    driver: Any,
    db: str,
    handler: str,
    seeds: list[str],
    query: str = "",
) -> list[str]:
    """多个标准入口各查一次，被更多入口同时召回的目标排前面。"""
    names = [item for item in seeds if item][:4]
    if len(names) < 2:
        return []
    history = _user_finding_blob(driver, db)
    scored: list[tuple[float, str]] = []
    if handler in {"kg_symptom_parts", "kg_symptom_care"}:
        rows = _query(
            driver,
            db,
            """
            UNWIND $seeds AS sym
            MATCH (s:ConsultPrimarySymptom)-[:BELONG|Belong]-(d:ConsultDisease)-[:BELONG|Belong]-(p:ConsultBodyPart)
            WHERE s.symptom_name = sym OR s.symptom_name CONTAINS sym OR sym CONTAINS s.symptom_name
            RETURN sym AS seed, p.part_name AS target, d.disease_name AS via
            LIMIT 40
            """,
            {"seeds": names},
        )
        grouped: dict[str, set[str]] = {}
        via: dict[str, str] = {}
        for row in rows:
            target = str(row.get("target") or "").strip()
            seed = str(row.get("seed") or "").strip()
            if not target or not seed:
                continue
            grouped.setdefault(target, set()).add(seed)
            if row.get("via") and target not in via:
                via[target] = str(row.get("via") or "").strip()
        for target, got in grouped.items():
            line = f"{'/'.join(sorted(got))}—{target}" + (f"—{via[target]}" if via.get(target) else "")
            score = float(len(got))
            if history and target in history:
                score += 2.0
                line = "与既往记录重合：" + line
            scored.append((score, line))
    elif handler == "kg_part_diseases":
        rows = _query(
            driver,
            db,
            """
            UNWIND $seeds AS part_name
            MATCH (p:ConsultBodyPart)-[:BELONG|Belong]-(d:ConsultDisease)
            WHERE p.part_name = part_name OR p.part_name CONTAINS part_name OR part_name CONTAINS p.part_name
            RETURN part_name AS seed, d.disease_name AS target
            LIMIT 40
            """,
            {"seeds": names},
        )
        grouped = {}
        for row in rows:
            target = str(row.get("target") or "").strip()
            seed = str(row.get("seed") or "").strip()
            if target and seed:
                grouped.setdefault(target, set()).add(seed)
        for target, got in grouped.items():
            line = f"{'/'.join(sorted(got))}—{target}"
            score = float(len(got))
            if history and target in history:
                score += 2.0
                line = "与既往记录重合：" + line
            scored.append((score, line))
    elif handler in {"kg_disease_symptoms", "kg_disease_plain"}:
        rows = _query(
            driver,
            db,
            """
            UNWIND $seeds AS dis
            MATCH (d:ConsultDisease)-[:BELONG|Belong]-(s:ConsultPrimarySymptom)
            WHERE d.disease_name = dis OR d.disease_name CONTAINS dis OR dis CONTAINS d.disease_name
            RETURN dis AS seed, s.symptom_name AS target
            LIMIT 40
            """,
            {"seeds": names},
        )
        grouped = {}
        for row in rows:
            target = str(row.get("target") or "").strip()
            seed = str(row.get("seed") or "").strip()
            if target and seed:
                grouped.setdefault(target, set()).add(seed)
        for target, got in grouped.items():
            line = f"{'/'.join(sorted(got))}—{target}"
            score = float(len(got))
            if history and any(name in history for name in got):
                score += 2.0
                line = "与既往记录重合：" + line
            scored.append((score, line))
    elif handler == "kg_syndrome_signs":
        rows = _query(
            driver,
            db,
            """
            UNWIND $seeds AS syn
            MATCH (st:SyndromeType)-[:BELONG|Belong]-(d:ConsultDisease)-[:BELONG|Belong]-(s:ConsultPrimarySymptom)
            WHERE st.main_symptoms CONTAINS syn OR d.disease_name CONTAINS syn
            RETURN syn AS seed, s.symptom_name AS target
            LIMIT 40
            """,
            {"seeds": names},
        )
        grouped = {}
        for row in rows:
            target = str(row.get("target") or "").strip()
            seed = str(row.get("seed") or "").strip()
            if target and seed:
                grouped.setdefault(target, set()).add(seed)
        for target, got in grouped.items():
            line = f"{'/'.join(sorted(got))}—{target}"
            score = float(len(got))
            scored.append((score, line))
    return _blend_path_scores(query, scored)


def _ranked_business_paths(
    driver: Any,
    db: str,
    mentions: dict[str, list[str]],
) -> list[str]:
    """检查或指标名落到病/部位，以及和用户 Finding 重合的路径，只留前 3 条。"""
    exams = _mention_list(mentions, "指标") + _mention_list(mentions, "检查")
    if not exams:
        return []
    history = _user_finding_blob(driver, db)
    scored: list[tuple[float, str]] = []
    for exam in exams[:3]:
        rows = _query(
            driver,
            db,
            """
            MATCH (d:ConsultDisease)-[:BELONG|Belong]-(p:ConsultBodyPart)
            WHERE d.disease_name CONTAINS $exam OR p.part_name CONTAINS $exam
            RETURN d.disease_name AS disease, p.part_name AS part
            LIMIT 12
            """,
            {"exam": exam[:40]},
        )
        for row in rows:
            disease = str(row.get("disease") or "").strip()
            part = str(row.get("part") or "").strip()
            if not disease:
                continue
            line = f"{exam}—{disease}—{part}"
            score = 1.5
            if history and (disease in history or (part and part in history)):
                score += 2.0
                line = "与既往记录重合：" + line
            scored.append((score, line))
    scored.sort(key=lambda item: -item[0])
    out: list[str] = []
    for _score, line in scored:
        if line not in out:
            out.append(line)
        if len(out) >= 3:
            break
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


def _aligned_names(alignments: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for hit in alignments or []:
        if not isinstance(hit, dict):
            continue
        name = str(hit.get("standard_name") or hit.get("mention") or "").strip()
        if name:
            names.append(name)
    return list(dict.fromkeys(names))[:8]


def _prefer_aligned_items(items: list[str], names: list[str]) -> list[str]:
    """有标准名时只留正文里出现这些名字的片段；一条都对不上则保留原结果。"""
    rows = [str(item).strip() for item in items if str(item).strip()]
    if not names:
        return rows
    kept = [row for row in rows if any(name in row for name in names)]
    return kept or rows


def _one_hop_lines(driver: Any, db: str, seeds: list[str]) -> list[str]:
    """对齐实体在问诊图上走一跳，写成「左—右」短句。"""
    lines: list[str] = []
    for seed in seeds[:4]:
        token = (seed or "").strip()
        if len(token) < 2:
            continue
        try:
            rows = _query(
                driver,
                db,
                """
                MATCH (a)-[:BELONG|Belong]-(b)
                WHERE coalesce(a.part_name, a.disease_name, a.symptom_name, '') = $seed
                   OR coalesce(a.part_name, a.disease_name, a.symptom_name, '') CONTAINS $seed
                RETURN coalesce(a.part_name, a.disease_name, a.symptom_name, '') AS left_name,
                       coalesce(b.part_name, b.disease_name, b.symptom_name, '') AS right_name
                LIMIT 6
                """,
                {"seed": token[:40]},
            )
        except Exception:
            logger.debug("one-hop skipped seed=%s", token, exc_info=True)
            continue
        for row in rows:
            left = str(row.get("left_name") or "").strip()
            right = str(row.get("right_name") or "").strip()
            if left and right and left != right:
                lines.append(f"{left}—{right}")
    return list(dict.fromkeys(lines))[:8]


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
