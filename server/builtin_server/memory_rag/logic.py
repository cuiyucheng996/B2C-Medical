"""个人记忆页：时间线在本服务拼装；读库与检索走陪聊内部接口。"""

from __future__ import annotations

import json
from typing import Any

from configs.paths_b2c import B2C_ROOT, ensure_b2c_runtime_path

ensure_b2c_runtime_path()

_MEMORY_TOOLS = ("query_personal_memory",)


def list_recent_user_ids(limit: int = 30) -> list[str]:
    """从 data/ui_drafts 读取最近用过的 user_id（可选参考）。"""
    root = B2C_ROOT / "data" / "user_production" / "ui_drafts"
    if not root.is_dir():
        return []
    rows: list[tuple[float, str]] = []
    for path in root.glob("*.json"):
        try:
            mtime = path.stat().st_mtime
            raw = json.loads(path.read_text(encoding="utf-8"))
            uid = str(raw.get("user_id") or "").strip() if isinstance(raw, dict) else ""
            if not uid:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        if uid:
            rows.append((mtime, uid))
    rows.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for _, uid in rows:
        if uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
        if len(out) >= limit:
            break
    return out


def load_card(user_id: str, *, limit: int = 30) -> dict[str, Any]:
    uid = (user_id or "").strip()
    if not uid:
        return {"ok": False, "error": "缺少 user_id"}
    from server.builtin_server.memory_rag.companion_client import companion_post

    remote = companion_post("/internal/memory/card", {"user_id": uid, "limit": limit})
    if not remote.get("ok"):
        return remote
    card = remote.get("card") if isinstance(remote.get("card"), dict) else {}
    return {"ok": True, "user_id": uid, "card": card, "timeline": build_timeline(card)}


def _kind_label(kind: str) -> str:
    mapping = {
        "body_part": "部位",
        "disease": "疾病",
        "symptom": "症状",
        "time": "时间",
        "place": "地点",
        "person": "人物",
        "event": "事件",
        "mood": "心情",
        "weather": "天气",
        "occupation": "职业",
        "habit": "习惯",
        "thing": "事物",
        "negation": "否定",
        "other": "其他",
    }
    return mapping.get((kind or "").strip(), kind or "实体")


def _finding_kind_label(kind: str) -> str:
    mapping = {
        "text": "文本原话",
        "image": "图片",
        "consult_state": "问诊状态快照",
    }
    return mapping.get((kind or "").strip(), kind or "Finding")


def _graph_layer(
    *,
    hops: int,
    rel: str,
    domain: str,
    kind: str,
    name: str,
    neo4j_label: str = "",
    mention: str = "",
    aligned: bool = False,
    negated: bool = False,
    graphiti_type: str = "",
) -> dict[str, Any]:
    kind_label = _kind_label(kind)
    target = neo4j_label or ("CompanionFact" if domain == "生活" and rel == "HAS_FACT" else "Knowledge")
    path = f"Finding —{rel}→ {target}" if rel != "STATED_FACT" else f"UserInfo —{rel}→ CompanionFact"
    return {
        "hops": hops,
        "rel": rel,
        "domain": domain,
        "kind": kind,
        "kind_label": kind_label,
        "neo4j_label": neo4j_label,
        "name": name,
        "mention": mention or name,
        "aligned": aligned,
        "negated": negated,
        "graphiti_type": (graphiti_type or "").strip(),
        "path": path,
    }


def _graph_for_finding(finding: dict[str, Any]) -> dict[str, Any]:
    fkind = str(finding.get("kind") or "text")
    memory_date = str(finding.get("memory_date") or "").strip()
    created = str(finding.get("created_at") or "").strip()
    if not memory_date and len(created) >= 10:
        memory_date = created[:10]
    finding_anchor: dict[str, Any] | None = None
    if memory_date:
        root = {
            "hops": 0,
            "node_label": "MemoryDay",
            "rel": None,
            "name": memory_date,
            "detail": "日历日 · Asia/Shanghai",
            "path": "UserInfo —ON_DAY→ MemoryDay",
        }
        finding_anchor = {
            "hops": 1,
            "node_label": "Finding",
            "rel": "HAS_FINDING",
            "name": _finding_kind_label(fkind),
            "detail": fkind,
            "path": "MemoryDay —HAS_FINDING→ Finding",
        }
        entity_hops = 2
    else:
        root = {
            "hops": 0,
            "node_label": "Finding",
            "rel": None,
            "name": _finding_kind_label(fkind),
            "detail": fkind,
            "path": "UserInfo —HAVE→ Finding",
        }
        entity_hops = 1
    layers: list[dict[str, Any]] = []
    for ex in finding.get("extracts") or []:
        if not isinstance(ex, dict):
            continue
        kind = str(ex.get("kind") or "other")
        name = str(ex.get("name") or ex.get("mention") or "").strip()
        if not name:
            continue
        layers.append(
            _graph_layer(
                hops=entity_hops,
                rel="EXTRACTED",
                domain="病理",
                kind=kind,
                name=name,
                neo4j_label=str(ex.get("label") or ""),
                mention=str(ex.get("mention") or name),
                aligned=bool(ex.get("aligned")),
            )
        )
    for lf in finding.get("linked_facts") or []:
        if not isinstance(lf, dict):
            continue
        name = str(lf.get("name") or "").strip()
        if not name:
            continue
        layers.append(
            _graph_layer(
                hops=entity_hops,
                rel="HAS_FACT",
                domain="生活",
                kind=str(lf.get("kind") or "other"),
                name=name,
                neo4j_label="CompanionFact",
                mention=str(lf.get("fact") or name),
                negated=bool(lf.get("negated")),
                graphiti_type=str(lf.get("graphiti_type") or ""),
            )
        )
    by_domain: dict[str, list[dict[str, Any]]] = {"病理": [], "生活": []}
    for layer in layers:
        by_domain[layer["domain"]].append(layer)
    return {
        "root": root,
        "finding_anchor": finding_anchor,
        "layers": layers,
        "by_domain": by_domain,
        "pathology_count": len(by_domain["病理"]),
        "life_count": len(by_domain["生活"]),
    }


def _graph_for_companion_fact(fact: dict[str, Any]) -> dict[str, Any]:
    kind = str(fact.get("kind") or "other")
    name = str(fact.get("name") or "").strip()
    root = {
        "hops": 0,
        "node_label": "CompanionFact",
        "rel": None,
        "name": name or str(fact.get("fact") or ""),
        "detail": kind,
        "path": "CompanionFact（生活实体）",
    }
    layers: list[dict[str, Any]] = [
        _graph_layer(
            hops=1,
            rel="STATED_FACT",
            domain="生活",
            kind=kind,
            name=name,
            neo4j_label="UserInfo",
            mention="当前用户",
            negated=bool(fact.get("negated")),
            graphiti_type=str(fact.get("graphiti_type") or ""),
        ),
    ]
    fid = str(fact.get("finding_id") or "").strip()
    if fid:
        layers.append(
            {
                "hops": 2,
                "rel": "HAVE+HAS_FACT",
                "domain": "生活",
                "kind": kind,
                "kind_label": _kind_label(kind),
                "neo4j_label": "Finding",
                "name": f"关联 Finding …{fid[-8:]}" if len(fid) > 8 else f"Finding {fid}",
                "mention": "UserInfo —HAVE→ Finding —HAS_FACT→ 本事实",
                "aligned": False,
                "negated": False,
                "path": "UserInfo —HAVE→ Finding —HAS_FACT→ CompanionFact",
            }
        )
    by_domain = {"病理": [], "生活": list(layers)}
    return {
        "root": root,
        "layers": layers,
        "by_domain": by_domain,
        "pathology_count": 0,
        "life_count": len(layers),
    }


def _snippet(text: str, n: int = 48) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def build_timeline(card: dict[str, Any]) -> dict[str, Any]:
    """把时间线块 + 实体片段整理成前端可直接渲染的结构。"""
    profile = {
        "summary": card.get("summary") or "",
        "body_part": card.get("current_body_part"),
        "disease": card.get("current_disease"),
        "symptoms": list(card.get("current_symptoms") or []),
    }
    blocks: list[dict[str, Any]] = []
    for finding in card.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        raw = str(finding.get("raw_text") or "").strip()
        created = str(finding.get("created_at") or finding.get("last_said") or "未知时间")
        memory_date = str(finding.get("memory_date") or "").strip()
        entities: list[dict[str, Any]] = []
        for ex in finding.get("extracts") or []:
            if not isinstance(ex, dict):
                continue
            kind = str(ex.get("kind") or "other")
            name = str(ex.get("name") or ex.get("mention") or "").strip()
            if not name:
                continue
            entities.append(
                {
                    "kind": kind,
                    "kind_label": _kind_label(kind),
                    "name": name,
                    "mention": str(ex.get("mention") or name),
                    "aligned": bool(ex.get("aligned")),
                    "node_label": str(ex.get("label") or ""),
                }
            )
        graph = _graph_for_finding(finding)
        blocks.append(
            {
                "id": str(finding.get("id") or ""),
                "type": "finding",
                "time": created,
                "memory_date": memory_date,
                "source": str(finding.get("source") or "chat"),
                "kind": str(finding.get("kind") or "text"),
                "title": _snippet(raw or "（无原文，仅有抽离）"),
                "raw_text": raw,
                "entity_count": len(entities),
                "entities": entities,
                "graph": graph,
                "pathology_count": graph["pathology_count"],
                "life_count": graph["life_count"],
            }
        )
    blocks.sort(key=lambda b: str(b.get("time") or ""), reverse=True)
    by_date: dict[str, list[dict[str, Any]]] = {}
    for block in blocks:
        md = str(block.get("memory_date") or "").strip()
        t = str(block.get("time") or "")
        date_key = md or (t[:10] if len(t) >= 10 else "未标注日期")
        by_date.setdefault(date_key, []).append(block)
    return {"profile": profile, "blocks": blocks, "by_date": by_date}


def search_memory(user_id: str, query: str, *, max_facts: int = 12) -> dict[str, Any]:
    uid = (user_id or "").strip()
    if not uid:
        return {"ok": False, "error": "缺少 user_id"}
    from server.builtin_server.memory_rag.companion_client import companion_post

    return companion_post(
        "/internal/memory/search",
        {"user_id": uid, "query": (query or "").strip(), "max_facts": max_facts},
    )


def rag_probe(user_id: str, query: str) -> dict[str, Any]:
    """检索探测交给陪聊服务，本进程不加载意图和工具。"""
    from server.builtin_server.memory_rag.companion_client import companion_post

    uid = (user_id or "").strip()
    q = (query or "").strip()
    if not uid:
        return {"ok": False, "error": "缺少 user_id"}
    if not q:
        return {"ok": False, "error": "缺少 query"}
    return companion_post(
        "/internal/memory/rag",
        {"user_id": uid, "query": q, "tools": list(_MEMORY_TOOLS)},
        timeout=180,
    )
