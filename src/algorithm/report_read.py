"""报告指标解读三件事：上传解读、口头提报告、指标对历史。"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "compare_lab_history",
    "explain_mentioned_report",
    "collect_report_facts",
    "explain_uploaded_report",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip()).lower()


def _mentions_list(mentions: dict[str, Any] | None, key: str) -> list[str]:
    row = mentions if isinstance(mentions, dict) else {}
    return [str(item).strip() for item in (row.get(key) or []) if str(item).strip()]


def _handbook(query: str, handler: str) -> dict[str, Any]:
    from algorithm.kg_query import search_chroma_knowledge, search_exam_handbook

    packed = search_exam_handbook(query=query, handler=handler)
    items = list(packed.get("items") or [])
    if items:
        return packed
    extra = search_chroma_knowledge(query=query, handler=handler)
    return extra if extra.get("items") else packed


def _facts(user_id: str, query: str) -> list[dict[str, Any]]:
    uid = (user_id or "").strip()
    q = (query or "").strip()
    if not uid or not q:
        return []
    from memory.graphiti_store import search_facts

    return list(search_facts(user_id=uid, query=q) or [])


def _match_indicators(indicators: list[dict[str, Any]], metric: str) -> list[dict[str, str]]:
    needle = _norm(metric)
    if not needle:
        return []
    hits: list[dict[str, str]] = []
    for item in indicators or []:
        name = str(item.get("name") or "")
        if needle in _norm(name) or _norm(name) in needle:
            hits.append(
                {
                    "name": name,
                    "value": str(item.get("value") or ""),
                    "unit": str(item.get("unit") or ""),
                    "flag": str(item.get("flag") or ""),
                }
            )
    return hits[:8]


def collect_report_facts(
    *,
    last_report: dict[str, Any] | None = None,
    last_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """抽指标逻辑在专属服务 server/builtin_server/report_read/logic.py。"""
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from server.builtin_server.report_read.logic import collect_report_facts as run

    return run(last_report=last_report, last_file=last_file)


def explain_uploaded_report(
    *,
    query: str = "",
    last_report: dict[str, Any] | None = None,
    last_file: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """预处理判定报告解读且给了 PDF：MinerU 拆该 PDF；否则只用切块抽指标。"""
    facts = collect_report_facts(last_report=last_report, last_file=last_file)
    if facts.get("empty"):
        return {
            "summary": "本轮没有上传检查报告",
            "report": {},
            "items": [],
            "empty": True,
        }
    names = list(facts.get("indicator_names") or [])
    filename = str(facts.get("filename") or "")
    q = (query or "").strip() or " ".join(names[:8]) or filename or "化验单"
    kb = _handbook(q[:200], "report_upload")
    items = list(kb.get("items") or [])
    bits = [f"已解读 {filename or '报告'}"]
    if facts.get("table_json_count"):
        bits.append(f"结构化表 {facts.get('table_json_count')} 张")
    elif facts.get("table_count"):
        bits.append(f"表区域 {facts.get('table_count')} 处")
    bits.append(f"抽出 {facts.get('indicator_count')} 条指标")
    if items:
        bits.append(f"手册命中 {len(items)} 条")
    return {
        "summary": "，".join(bits),
        "report": facts.get("report") or {},
        "items": items[:16],
        "kb_source": kb.get("source") or "",
        "empty": False,
    }


def explain_mentioned_report(
    *,
    exam: str = "",
    query: str = "",
    user_id: str = "",
    mentions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """用户口头提到的检查/化验单，查记忆 + 手册。"""
    exam_text = (exam or "").strip() or " ".join(_mentions_list(mentions, "检查"))
    q = " ".join(part for part in (exam_text, (query or "").strip()) if part).strip() or "检查报告"
    facts = _facts(user_id, q)
    kb = _handbook(q[:200], "report_explain")
    items = list(kb.get("items") or [])
    bits = []
    if exam_text:
        bits.append(f"检查：{exam_text}")
    if facts:
        bits.append(f"记忆 {len(facts)} 条")
    if items:
        bits.append(f"手册 {len(items)} 条")
    summary = "，".join(bits) or "没有找到这份检查的记忆或手册"
    return {
        "summary": summary,
        "exam": exam_text,
        "facts": facts[:12],
        "items": items[:16],
        "kb_source": kb.get("source") or "",
        "empty": not facts and not items,
    }


def compare_lab_history(
    *,
    metric: str = "",
    query: str = "",
    user_id: str = "",
    last_report: dict[str, Any] | None = None,
    mentions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """当前报告里的某指标 vs 长期记忆里同名记录。"""
    metric_text = (metric or "").strip() or " ".join(_mentions_list(mentions, "指标"))
    if not metric_text:
        # 从「对比一下胆固醇」这类话里抠后半截当指标名
        text = re.sub(r"(对比|比较|和上次|跟上次|历史|变化)", "", query or "")
        text = re.sub(r"(一下|看看|的|了)", "", text).strip()
        metric_text = text[:20]
    report = last_report if isinstance(last_report, dict) else {}
    current = _match_indicators(list(report.get("indicators") or []), metric_text)
    facts = _facts(user_id, metric_text or (query or ""))
    kb = _handbook((metric_text or query or "")[:200], "report_compare")
    items = list(kb.get("items") or [])
    if not metric_text:
        return {
            "summary": "没有指出要对比哪项指标",
            "metric": "",
            "current": [],
            "facts": [],
            "items": [],
            "empty": True,
            "need_metric": True,
        }
    bits = [f"指标：{metric_text}"]
    if current:
        bits.append(f"本轮 {len(current)} 条")
    if facts:
        bits.append(f"历史 {len(facts)} 条")
    if items:
        bits.append("含参考范围")
    return {
        "summary": "，".join(bits),
        "metric": metric_text,
        "current": current,
        "facts": facts[:12],
        "items": items[:12],
        "kb_source": kb.get("source") or "",
        "empty": not current and not facts,
        "need_metric": False,
    }
