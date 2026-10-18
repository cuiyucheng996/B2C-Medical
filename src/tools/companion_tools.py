"""按细意图装配的陪聊工具：只回 JSON，不对用户说话。"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from langchain_core.tools import BaseTool, tool

from tools.companion_memory import companion_last_facts, companion_parsed_report, companion_user_id
from tools.tool_envelope import dump_envelope

__all__ = [
    "companion_turn",
    "lookup_stated_consult",
    "lookup_life_facts",
    "query_personal_memory",
    "query_consult_graph",
    "search_exam_handbook",
    "search_chroma_knowledge",
    "note_task_remind",
    "note_task_checkin",
    "note_task_medication",
    "note_task_appointment",
    "note_task_report",
    "start_lab_report_sync",
    "poll_lab_report_sync",
    "explain_uploaded_report",
    "extract_report_tables",
    "explain_mentioned_report",
    "compare_lab_history",
    "describe_uploaded_image",
    "describe_uploaded_document",
    "safety_policy_hint",
    "tools_for_intent",
    "mark_companion_tools",
]

# 具体任务 kind → 给用户看的中文名。kind 是 n8n 主流程 Switch 的分支键，两端必须同改
TASK_KINDS: dict[str, str] = {
    "remind": "复查提醒",
    "checkin": "打卡",
    "medication": "服药提醒",
    "appointment": "复诊预约",
    "report": "取报告提醒",
}

companion_turn: ContextVar[dict[str, Any]] = ContextVar("companion_turn", default={})

_LIFE_KINDS = frozenset(
    {"place", "time", "mood", "weather", "event", "occupation", "habit", "person", "thing"}
)


def _life_fact_query_needles(query: str) -> list[str]:
    import re

    text = (query or "").strip()
    needles: list[str] = []
    if text:
        needles.append(text)
    for m in re.finditer(r"(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})", text):
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        needles.extend([f"{y}-{mo:02d}-{d:02d}", f"{mo}月{d}日", f"{mo}月{d}", f"{mo}月{d}号"])
    m = re.search(r"(\d{1,2})月(\d{1,2})", text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        needles.extend([f"-{mo:02d}-{d:02d}", f"{mo}月{d}日", f"{mo}月{d}", f"{mo}月{d}号"])
    out: list[str] = []
    for n in needles:
        n = (n or "").strip()
        if n and n not in out:
            out.append(n)
    return out


def _life_fact_blob(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("kind", "name", "fact", "last_said")
    )


def _turn() -> dict[str, Any]:
    row = companion_turn.get()
    return dict(row) if isinstance(row, dict) else {}


@tool
def lookup_stated_consult() -> str:
    """查询用户曾经提到的部位、疾病、症状（当前画像 + Finding）。不问心情事件。"""
    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    if not uid:
        return dump_envelope(
            tool="lookup_stated_consult",
            ok=False,
            error="无 user_id",
            usable=False,
            retryable=False,
        )
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.algorithm.user_memory import load_user_memory_card

    card = load_user_memory_card(uid, limit=8)
    extracts: list[dict[str, Any]] = []
    for finding in card.get("findings") or []:
        for item in finding.get("extracts") or []:
            kind = str(item.get("kind") or "")
            if kind not in {"body_part", "disease", "symptom"}:
                continue
            extracts.append(
                {
                    "kind": kind,
                    "name": item.get("name") or item.get("mention"),
                    "raw": str(finding.get("raw_text") or "")[:80],
                }
            )
    payload = {
        "current_body_part": card.get("current_body_part"),
        "current_disease": card.get("current_disease"),
        "current_symptoms": list(card.get("current_symptoms") or []),
        "extracts": extracts[:12],
        "summary": card.get("summary") or "",
    }
    companion_last_facts.set(
        [
            {"name": "stated_consult", "fact": payload.get("summary") or "无问诊向记忆"},
        ]
    )
    return dump_envelope(
        tool="lookup_stated_consult",
        summary=str(payload.get("summary") or ""),
        data=payload,
    )


@tool
def lookup_life_facts(query: str = "") -> str:
    """查询用户最近的心情、事件、地点、时间、随口提到的人/事物。不查部位疾病画像。"""
    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    if not uid:
        return dump_envelope(
            tool="lookup_life_facts",
            ok=False,
            error="无 user_id",
            usable=False,
            retryable=False,
        )
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.knowledge.user_finding import list_companion_facts

    q = (query or "").strip()
    needles = _life_fact_query_needles(q)
    fetch_limit = 120 if needles else 20
    rows: list[dict[str, Any]] = []
    for item in list_companion_facts(uid, limit=fetch_limit):
        kind = str(item.get("kind") or "")
        if kind not in _LIFE_KINDS:
            continue
        if needles and not any(n in _life_fact_blob(item) for n in needles):
            continue
        name = str(item.get("name") or "").strip()
        fact = str(item.get("fact") or "").strip()
        rows.append(
            {
                "kind": kind,
                "name": name,
                "fact": ("已排除：" + name) if item.get("negated") else fact,
                "last_said": item.get("last_said") or "",
            }
        )
    if not rows and needles:
        from memory.graphiti_store import search_facts

        for item in search_facts(user_id=uid, query=q, max_facts=12):
            fact = str(item.get("fact") or "").strip()
            if not fact:
                continue
            rows.append(
                {
                    "kind": str(item.get("name") or "memory"),
                    "name": str(item.get("name") or ""),
                    "fact": fact,
                    "last_said": "",
                }
            )
    companion_last_facts.set(rows)
    if not rows:
        return dump_envelope(
            tool="lookup_life_facts",
            summary="暂无心情/事件类事实",
            data={"facts": []},
            usable=False,
            retryable=True,
        )
    return dump_envelope(tool="lookup_life_facts", summary=f"{len(rows[:12])} 条生活事实", data={"facts": rows[:12]})


@tool
def query_personal_memory(
    mode: str,
    entity_domain: str = "life",
    entity_kinds: str = "",
    time_start: str = "",
    time_end: str = "",
    time_hint: str = "",
    entity_hint: str = "",
    keywords: str = "",
) -> str:
    """按时间或实体入口查个人记忆库（Neo4j Finding + CompanionFact）。

    mode 必填：time_to_entity（先定时间再找实体）或 entity_to_time（先定实体再找时间）。
    entity_domain：pathology（部位 body_part / 疾病 disease / 症状 symptom）或 life（心情 mood、事物 thing、地点 place、事件 event 等）。
    entity_kinds：逗号分隔，须与 domain 一致，如 thing,mood 或 body_part,symptom。
    time_to_entity 时填 time_start/time_end（YYYY-MM-DD）或 time_hint（如「前些天晚上」）；可填 keywords 辅助匹配原文。
    entity_to_time 时填 entity_hint（如「胃疼」「作业」）与 entity_kinds。
    """
    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    from algorithm.memory_query import query_personal_memory as run

    payload = run(
        user_id=uid,
        mode=mode,
        entity_domain=entity_domain,
        entity_kinds=entity_kinds,
        time_start=time_start,
        time_end=time_end,
        time_hint=time_hint,
        entity_hint=entity_hint,
        keywords=keywords,
    )
    hits = payload.get("hits") or []
    companion_last_facts.set(
        [
            {
                "name": str(item.get("name") or item.get("kind") or "memory"),
                "fact": str(item.get("fact") or item.get("raw_text") or ""),
            }
            for item in hits[:12]
            if isinstance(item, dict)
        ]
    )
    if not payload.get("ok"):
        return dump_envelope(
            tool="query_personal_memory",
            ok=False,
            error=str(payload.get("error") or "查询失败"),
            data=payload,
            usable=False,
            retryable=False,
        )
    if not hits:
        return dump_envelope(
            tool="query_personal_memory",
            summary="未命中个人记忆",
            data=payload,
            usable=False,
            retryable=True,
        )
    return dump_envelope(
        tool="query_personal_memory",
        summary=f"命中 {len(hits)} 条",
        data=payload,
    )


def _kg_turn_query(query: str) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    turn = _turn()
    name = str(turn.get("handler") or "fallback")
    text = (query or "").strip() or str(turn.get("query") or "")
    mentions = turn.get("mentions") if isinstance(turn.get("mentions"), dict) else {}
    alignments = turn.get("alignments") if isinstance(turn.get("alignments"), list) else []
    return name, text, mentions, alignments


def _kg_dump(tool: str, result: dict[str, Any]) -> str:
    data: dict[str, Any] = result
    summary = str(result.get("summary") or "")
    if tool == "search_exam_handbook":
        from algorithm.kg_query import slim_handbook_pack_for_agent

        data = slim_handbook_pack_for_agent(result)
        if not summary:
            summary = str(data.get("summary") or "")
    return dump_envelope(
        tool=tool,
        summary=summary,
        data=data,
    )


@tool
def query_consult_graph(query: str = "") -> str:
    """只查 Neo4j 问诊图（部位-病、病-症、部位、通俗解释、证型）。不要编清单。"""
    from algorithm.kg_query import query_consult_graph as run

    name, text, mentions, alignments = _kg_turn_query(query)
    return _kg_dump(
        "query_consult_graph",
        run(handler=name, query=text, mentions=mentions, alignments=alignments),
    )


@tool
def search_exam_handbook(query: str = "") -> str:
    """只查 RAGFlow / 本地检查手册 PDF 分块。不要编检查含义。"""
    from algorithm.kg_query import search_exam_handbook as run

    name, text, mentions, alignments = _kg_turn_query(query)
    return _kg_dump(
        "search_exam_handbook",
        run(query=text, mentions=mentions, alignments=alignments, handler=name),
    )


@tool
def search_chroma_knowledge(query: str = "") -> str:
    """只查 Chroma 疾病/症状向量库。不要编症状清单。"""
    from algorithm.kg_query import search_chroma_knowledge as run

    _name, text, mentions, alignments = _kg_turn_query(query)
    extra = []
    for key in ("部位", "疾病", "症状", "检查", "指标"):
        extra.extend(mentions.get(key) or [])
    q = " ".join(part for part in [text, *extra] if part).strip() or text
    return _kg_dump(
        "search_chroma_knowledge",
        run(query=q, handler=_name, alignments=alignments),
    )


def _report_dump(tool: str, payload: dict[str, Any], *, retryable: bool = True) -> str:
    empty = bool(payload.get("empty"))
    return dump_envelope(
        tool=tool,
        summary=str(payload.get("summary") or ""),
        data=payload,
        usable=not empty,
        retryable=retryable and empty,
    )


def _last_report() -> dict[str, Any]:
    row = _turn().get("last_report")
    return dict(row) if isinstance(row, dict) else {}


@tool
def explain_uploaded_report(query: str = "") -> str:
    """MinerU 拆表（进程内按文件 SHA 缓存）；命中则跳过 GPU；否则用切块抽指标。"""
    from algorithm.report_read import explain_uploaded_report as run

    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    q = (query or "").strip() or str(_turn().get("query") or "")
    payload = run(query=q, last_report=_last_report(), last_file=_last_file())
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    if report:
        companion_last_facts.set(
            [{"name": "uploaded_report", "fact": payload.get("summary") or "", "user_id": uid}]
        )
        companion_parsed_report.set(report)
    return _report_dump("explain_uploaded_report", payload, retryable=False)


@tool
def extract_report_tables(path: str = "") -> str:
    """用 UnstructuredPDFLoader hi_res（YOLOX + OCR + infer_table_structure）抽表格行列 HTML。"""
    from algorithm.unstructured_tables import extract_tables_unstructured

    row = _last_file()
    target = (path or "").strip() or str(row.get("mineru_pdf_path") or row.get("saved_path") or "")
    if not target:
        return dump_envelope(
            tool="extract_report_tables",
            summary="没有可解析的 PDF/图片路径",
            data={"file": row},
            usable=False,
            retryable=True,
        )
    packed = extract_tables_unstructured(target)
    payload = {
        "summary": f"抽到 {packed.get('table_count') or 0} 张表（Unstructured hi_res）",
        "table_count": packed.get("table_count") or 0,
        "tables": list(packed.get("tables") or [])[:12],
        "preprocess": packed.get("preprocess") or {},
        "error": packed.get("error") or "",
        "empty": not packed.get("ok"),
    }
    return _report_dump("extract_report_tables", payload, retryable=not packed.get("ok"))


@tool
def explain_mentioned_report(exam: str = "") -> str:
    """解读用户口头提到的检查/化验单（查记忆和手册）。不处理刚上传的 PDF、也不做指标对比。exam 为检查名。"""
    from algorithm.report_read import explain_mentioned_report as run

    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    hitl = _hitl_answer()
    exam_text = (exam or "").strip() or hitl
    mentions = _turn().get("mentions") if isinstance(_turn().get("mentions"), dict) else {}
    payload = run(
        exam=exam_text,
        query=str(_turn().get("query") or ""),
        user_id=uid,
        mentions=mentions,
    )
    companion_last_facts.set(list(payload.get("facts") or []))
    return _report_dump("explain_mentioned_report", payload)


@tool
def compare_lab_history(metric: str = "") -> str:
    """把某项指标和历史记录对比。只办对比；不解读整张上传单。metric 为指标名，如胆固醇、空腹血糖。"""
    from algorithm.report_read import compare_lab_history as run

    uid = companion_user_id.get() or str(_turn().get("user_id") or "")
    hitl = _hitl_answer()
    metric_text = (metric or "").strip() or hitl
    mentions = _turn().get("mentions") if isinstance(_turn().get("mentions"), dict) else {}
    payload = run(
        metric=metric_text,
        query=str(_turn().get("query") or ""),
        user_id=uid,
        last_report=_last_report(),
        mentions=mentions,
    )
    companion_last_facts.set(list(payload.get("facts") or []))
    return _report_dump("compare_lab_history", payload, retryable=bool(payload.get("need_metric")))


@tool
def diagnose_tongue(query: str = "") -> str:
    """对本轮已判定为舌头的图片做舌象分析。返回舌色、舌型、苔色、苔质，不是诊断。"""
    row = dict(_last_file())
    path = str(row.get("saved_path") or "")
    if str(row.get("role") or "") != "tongue_image" or not path:
        return dump_envelope(
            tool="diagnose_tongue",
            summary="本轮没有检出舌头的图片",
            data={"file": row},
            usable=False,
            retryable=False,
        )
    from algorithm.tongue_client import diagnose_tongue_file

    packed = diagnose_tongue_file(path)
    data = packed.get("data") if isinstance(packed.get("data"), dict) else {}
    if not packed.get("ok"):
        return dump_envelope(
            tool="diagnose_tongue",
            summary=str(packed.get("msg") or packed.get("error") or "舌诊失败"),
            data=packed,
            usable=False,
            retryable=True,
        )
    summary = "，".join(
        f"{key}={value}"
        for key, value in (
            ("舌色", data.get("tongueColour")),
            ("舌型", data.get("tongueShape")),
            ("苔色", data.get("tongueFurColour")),
            ("苔质", data.get("tongueFurQuality")),
        )
        if value
    )
    return dump_envelope(
        tool="diagnose_tongue",
        summary=summary or "已返回舌象",
        data={"query": query, "tongue": data},
        usable=True,
        retryable=False,
    )


def _last_file() -> dict[str, Any]:
    row = _turn().get("last_file")
    return dict(row) if isinstance(row, dict) else {}


@tool
def describe_uploaded_image(query: str = "") -> str:
    """说明本轮上传的日常图片。没有抽过字时先按原件做版面识别。"""
    row = dict(_last_file())
    if str(row.get("family") or "") != "image" or str(row.get("role") or "") == "lab_report":
        return dump_envelope(
            tool="describe_uploaded_image",
            summary="本轮没有日常图片",
            data={"file": row},
            usable=False,
            retryable=False,
        )
    ocr = str(row.get("user_text") or row.get("markdown_preview") or "").strip()
    if row.get("saved_path") and (not ocr or "尚未" in ocr or row.get("parse_source") == "preview"):
        from algorithm.file_gate import parse_user_file

        parsed = parse_user_file(
            saved_path=str(row.get("saved_path") or ""),
            filename=str(row.get("filename") or ""),
            family="image",
        )
        row = parsed.to_dict()
        ocr = str(parsed.user_text or parsed.markdown or "").strip()
    payload = {
        "summary": f"图片 {row.get('filename') or ''} 已识别文字摘录",
        "file": {
            "filename": row.get("filename") or "",
            "family": row.get("family"),
            "role": row.get("role"),
            "reasons": list(row.get("reasons") or []),
            "markdown_preview": str(row.get("markdown_preview") or "")[:800],
        },
        "ocr": ocr[:2500],
        "query": (query or "").strip() or str(_turn().get("query") or ""),
        "empty": not ocr,
    }
    return _report_dump("describe_uploaded_image", payload, retryable=False)


@tool
def describe_uploaded_document(query: str = "") -> str:
    """说明本轮上传的普通文档（Word/Excel/PDF 等，非化验单）。不处理图片、不解读指标。"""
    row = _last_file()
    if str(row.get("family") or "") != "document" or str(row.get("role") or "") == "lab_report":
        return dump_envelope(
            tool="describe_uploaded_document",
            summary="本轮没有普通文档",
            data={"file": row},
            usable=False,
            retryable=False,
        )
    ocr = str(row.get("user_text") or row.get("markdown_preview") or "").strip()
    if row.get("saved_path") and (not ocr or "尚未" in ocr or row.get("parse_source") == "preview"):
        from algorithm.file_gate import parse_user_file

        parsed = parse_user_file(
            saved_path=str(row.get("saved_path") or ""),
            filename=str(row.get("filename") or ""),
            family="document",
        )
        row = parsed.to_dict()
        ocr = str(parsed.user_text or parsed.markdown or "").strip()
    payload = {
        "summary": f"文档 {row.get('filename') or ''} 已抽出正文",
        "file": {
            "filename": row.get("filename") or "",
            "family": row.get("family"),
            "role": row.get("role"),
            "reasons": list(row.get("reasons") or []),
            "markdown_preview": str(row.get("markdown_preview") or "")[:800],
        },
        "ocr": ocr[:2500],
        "query": (query or "").strip() or str(_turn().get("query") or ""),
        "empty": not ocr,
    }
    return _report_dump("describe_uploaded_document", payload, retryable=False)


def _hitl_answer() -> str:
    hitl = _turn().get("hitl") if isinstance(_turn().get("hitl"), dict) else {}
    return str(hitl.get("answer") or "")


def _push_n8n(*, kind: str, title: str, when_text: str = "", start_iso: str = "") -> dict[str, Any]:
    turn = _turn()
    from algorithm.n8n_task import push_companion_task

    uid = companion_user_id.get() or str(turn.get("user_id") or "")
    return push_companion_task(
        kind=kind,
        user_id=uid,
        session_id=str(turn.get("session_id") or ""),
        title=title,
        when_text=when_text,
        query=str(turn.get("query") or ""),
        start_iso=start_iso,
    )


def _task_note(
    *,
    kind: str,
    title: str,
    when_text: str = "",
    start_iso: str = "",
    hitl: str = "",
) -> tuple[str, dict[str, Any]]:
    """推 n8n，并把返回翻译成「给用户的一句话 + data」。五种具体任务共用。"""
    n8n = _push_n8n(kind=kind, title=title, when_text=when_text, start_iso=start_iso)
    label = TASK_KINDS.get(kind, "任务")
    if n8n.get("skipped"):
        note = f"已记下{label}（{title}），n8n 未配置，提醒未发出。"
    elif n8n.get("ok"):
        note = f"已记下{label}并提交提醒（{title}）。"
    else:
        note = f"已记下{label}，提醒发送失败：{n8n.get('error') or 'n8n 无响应'}。"
    data = {
        "kind": kind,
        "when": when_text,
        "start_iso": start_iso,
        "note": note,
        "answer": hitl,
        "n8n": n8n,
    }
    return note, data


@tool
def note_task_remind(when: str = "", start_iso: str = "") -> str:
    """记下到点提醒并推到 n8n。复查、赶飞机、办事都可以。不处理打卡、服药、复诊预约、取报告。when 为口语时间；有 Time MCP 结果时填 start_iso。"""
    hitl = _hitl_answer()
    when_text = (when or "").strip() or hitl
    query = str(_turn().get("query") or "")
    title = (when_text or query or "复查提醒")[:80]
    note, data = _task_note(kind="remind", title=title, when_text=when_text, start_iso=(start_iso or "").strip(), hitl=hitl)
    return dump_envelope(
        tool="note_task_remind",
        summary=note,
        data=data,
        usable=True,
        retryable=False,
    )


@tool
def note_task_checkin(what: str = "", start_iso: str = "") -> str:
    """记下打卡并推到 n8n 记账。不处理复查提醒、服药、复诊预约、取报告。what 为打卡内容。"""
    hitl = _hitl_answer()
    what_text = (what or "").strip() or hitl
    query = str(_turn().get("query") or "")
    title = (what_text or query or "健康打卡")[:80]
    note, data = _task_note(kind="checkin", title=title, when_text=what_text, start_iso=(start_iso or "").strip(), hitl=hitl)
    return dump_envelope(
        tool="note_task_checkin",
        summary=note,
        data=data,
        usable=True,
        retryable=False,
    )


@tool
def note_task_medication(drug: str = "", when: str = "", start_iso: str = "") -> str:
    """记下服药提醒并推到 n8n 定时发提醒。只办服药；不处理复查提醒、打卡、复诊预约、取报告。drug 为药名，when 为口语时间或频次。"""
    hitl = _hitl_answer()
    drug_text = (drug or "").strip()
    when_text = (when or "").strip() or hitl
    query = str(_turn().get("query") or "")
    title = "，".join(part for part in (drug_text, when_text) if part) or query or "服药提醒"
    note, data = _task_note(kind="medication", title=title[:80], when_text=when_text, start_iso=(start_iso or "").strip(), hitl=hitl)
    note_and_data = {"drug": drug_text, **data}
    return dump_envelope(
        tool="note_task_medication",
        summary=note,
        data=note_and_data,
        usable=True,
        retryable=False,
    )


@tool
def note_task_appointment(department: str = "", when: str = "", start_iso: str = "") -> str:
    """记下复诊预约并推到 n8n，到点前发提醒。只办复诊挂号；不处理复查提醒、打卡、服药、取报告。department 为医院或科室，when 为口语时间。"""
    hitl = _hitl_answer()
    dept_text = (department or "").strip()
    when_text = (when or "").strip() or hitl
    query = str(_turn().get("query") or "")
    title = "，".join(part for part in (dept_text, when_text) if part) or query or "复诊预约"
    note, data = _task_note(kind="appointment", title=title[:80], when_text=when_text, start_iso=(start_iso or "").strip(), hitl=hitl)
    return dump_envelope(
        tool="note_task_appointment",
        summary=note,
        data={"department": dept_text, **data},
        usable=True,
        retryable=False,
    )


@tool
def note_task_report(exam: str = "", when: str = "", start_iso: str = "") -> str:
    """记下取报告/回传报告提醒并推到 n8n 定时发提醒。只办取报告；不处理复查提醒、打卡、服药、复诊预约。exam 为哪项检查或报告，when 为口语时间。"""
    hitl = _hitl_answer()
    exam_text = (exam or "").strip()
    when_text = (when or "").strip() or hitl
    query = str(_turn().get("query") or "")
    title = "，".join(part for part in (exam_text, when_text) if part) or query or "取报告提醒"
    note, data = _task_note(kind="report", title=title[:80], when_text=when_text, start_iso=(start_iso or "").strip(), hitl=hitl)
    return dump_envelope(
        tool="note_task_report",
        summary=note,
        data={"exam": exam_text, **data},
        usable=True,
        retryable=False,
    )


from algorithm.lab_report_entities import LabReportSyncPollInput, LabReportSyncStartInput


@tool(args_schema=LabReportSyncStartInput)
def start_lab_report_sync(partner_id: str = "", exam_kind: str = "", when: str = "latest") -> str:
    """从已对接合作机构发起检验报告同步。partner_id、exam_kind、when 使用观察里已归一的实体，禁止自造。"""
    from algorithm.lab_report_sync_config import get_lab_report_sync_config, resolve_exam
    from algorithm.rpa_lab_job_store import create_job
    from algorithm.rpa_lab_runner import schedule_lab_report_sync

    cfg = get_lab_report_sync_config()
    if not cfg.enabled:
        return dump_envelope(
            tool="start_lab_report_sync",
            ok=False,
            error="报告同步未启用",
            usable=False,
            retryable=False,
        )
    turn = _turn()
    entities = turn.get("lab_sync_entities") if isinstance(turn.get("lab_sync_entities"), dict) else {}
    partner_id = (partner_id or str(entities.get("partner_id") or "")).strip()
    exam_kind = (exam_kind or str(entities.get("exam_kind") or "")).strip()
    when = (when or str(entities.get("when") or "latest")).strip() or "latest"
    if not partner_id or not exam_kind:
        missing = [name for name, val in (("机构", partner_id), ("检查", exam_kind)) if not val]
        return dump_envelope(
            tool="start_lab_report_sync",
            ok=False,
            error="缺少实体：" + "、".join(missing),
            usable=False,
            retryable=True,
            data={"entities": entities, "partners": list(cfg.partners.keys())},
        )
    resolved = resolve_exam(partner_id, exam_kind)
    if resolved is None:
        return dump_envelope(
            tool="start_lab_report_sync",
            ok=False,
            error=f"未知机构或检查类型：partner_id={partner_id!r} exam_kind={exam_kind!r}",
            usable=False,
            retryable=False,
            data={"partners": list(cfg.partners.keys())},
        )
    turn = _turn()
    uid = companion_user_id.get() or str(turn.get("user_id") or "")
    session_id = str(turn.get("session_id") or "")
    cfg.download_dir.mkdir(parents=True, exist_ok=True)
    job = create_job(
        user_id=uid,
        session_id=session_id,
        partner_id=resolved.partner_id,
        exam_kind=resolved.exam_kind,
        download_dir=cfg.download_dir,
        robot_app_uuid=resolved.robot_app_uuid,
    )
    schedule_lab_report_sync(job)
    summary = (
        f"已向{resolved.partner_title}发起{resolved.exam_label}报告同步（任务 {job.job_id[:8]}…）。"
        "若需登录/扫码，请看影刀窗口；稍后用 poll_lab_report_sync 查是否下载完成。"
    )
    return dump_envelope(
        tool="start_lab_report_sync",
        summary=summary,
        data={
            "job_id": job.job_id,
            "status": job.status,
            "partner_id": resolved.partner_id,
            "exam_kind": resolved.exam_kind,
            "when": when,
            "output_path": job.output_path,
        },
        usable=True,
        retryable=False,
    )


@tool(args_schema=LabReportSyncPollInput)
def poll_lab_report_sync(job_id: str = "") -> str:
    """查询检验报告同步任务是否完成；完成后自动接入报告解读。job_id 可空（用本会话最近一次任务）。"""
    from pathlib import Path

    from algorithm.rpa_lab_ingest import ingest_job_pdf
    from algorithm.rpa_lab_job_store import get_job, latest_job_for_session

    turn = _turn()
    jid = (job_id or "").strip()
    job = get_job(jid) if jid else latest_job_for_session(str(turn.get("session_id") or ""))
    if job is None:
        return dump_envelope(
            tool="poll_lab_report_sync",
            ok=False,
            error="没有可查询的同步任务",
            usable=False,
            retryable=True,
        )
    if job.status == "failed":
        return dump_envelope(
            tool="poll_lab_report_sync",
            summary=f"同步失败：{job.error or '未知错误'}",
            data=job.to_dict(),
            usable=False,
            retryable=True,
        )
    pdf = Path(job.output_path)
    if job.status == "done" or (pdf.is_file() and pdf.stat().st_size > 0):
        if job.status != "done":
            result = ingest_job_pdf(job)
        else:
            result = {"ok": True, "explain": {"summary": "报告已同步完成"}, "ingested": {}}
        if not result.get("ok"):
            return dump_envelope(
                tool="poll_lab_report_sync",
                ok=False,
                error=str(result.get("error") or "解读失败"),
                data=job.to_dict(),
                usable=False,
                retryable=True,
            )
        explain = result.get("explain") if isinstance(result.get("explain"), dict) else {}
        summary = str(explain.get("summary") or "报告已下载并完成解读。")
        ingested = result.get("ingested") if isinstance(result.get("ingested"), dict) else {}
        if ingested:
            companion_parsed_report.set(ingested.get("parsed") or {})
        return dump_envelope(
            tool="poll_lab_report_sync",
            summary=summary,
            data={"job": job.to_dict(), "explain": explain, "ingested": ingested},
            usable=True,
            retryable=False,
        )
    return dump_envelope(
        tool="poll_lab_report_sync",
        summary="报告仍在拉取中，请稍后再问；若需扫码请看影刀窗口。",
        data=job.to_dict(),
        usable=False,
        retryable=True,
    )


@tool
def safety_policy_hint() -> str:
    """急症或用药剂量：返回安全口径，禁止给出具体剂量或停药指导。"""
    handler = str(_turn().get("handler") or "")
    if handler == "risk_dosage":
        text = "不能提供具体用药剂量或停药方案，请咨询执业医师或药师。"
    else:
        text = "若有胸痛、呼吸困难、意识不清等急症表现，应立即就医或拨打急救电话。"
    return dump_envelope(
        tool="safety_policy_hint",
        summary=text,
        data={"policy": text},
        usable=True,
        retryable=False,
    )


def mark_companion_tools() -> None:
    """逻辑写在上面，这里只标域。"""
    from tools.companion_memory import search_long_memory
    from tools.tool_entry import mark_companion_tool

    if getattr(mark_companion_tools, "_done", False):
        return
    mark_companion_tool(safety_policy_hint, "safety")
    mark_companion_tool(query_consult_graph, "knowledge")
    mark_companion_tool(search_exam_handbook, "knowledge")
    mark_companion_tool(search_chroma_knowledge, "knowledge")
    mark_companion_tool(lookup_stated_consult, "memory", majors=("近况陈述",))
    mark_companion_tool(
        lookup_life_facts,
        "memory",
        majors=("近况陈述",),
    )
    mark_companion_tool(
        query_personal_memory,
        "memory",
        handlers=("mem_time_to_entity", "mem_entity_to_time"),
    )
    mark_companion_tool(search_long_memory, "memory", majors=("近况陈述",))
    mark_companion_tool(diagnose_tongue, "tongue", handlers=("tongue_recognize",))
    mark_companion_tool(describe_uploaded_image, "file", handlers=("file_image",))
    mark_companion_tool(describe_uploaded_document, "file", handlers=("file_document",))
    mark_companion_tool(explain_uploaded_report, "report", handlers=("report_upload",))
    mark_companion_tool(extract_report_tables, "report", handlers=("report_tables", "report_upload"))
    mark_companion_tool(explain_mentioned_report, "report", handlers=("report_explain",))
    mark_companion_tool(compare_lab_history, "report", handlers=("report_compare",))
    mark_companion_tool(start_lab_report_sync, "report", handlers=("report_sync",))
    mark_companion_tool(poll_lab_report_sync, "report", handlers=("report_sync",))
    mark_companion_tool(note_task_remind, "task", handlers=("task_remind",))
    mark_companion_tool(note_task_checkin, "task", handlers=("task_checkin",))
    mark_companion_tool(note_task_medication, "task", handlers=("task_medication",))
    mark_companion_tool(note_task_appointment, "task", handlers=("task_appointment",))
    mark_companion_tool(note_task_report, "task", handlers=("task_report",))
    mark_companion_tools._done = True  # type: ignore[attr-defined]


def tools_for_intent(*, major: str, handler: str) -> list[BaseTool]:
    """大类 + 细 handler 从注册表取本轮工具。"""
    from tools.registry import tools_from_registry

    return tools_from_registry(major=(major or "").strip(), handler=(handler or "").strip())
