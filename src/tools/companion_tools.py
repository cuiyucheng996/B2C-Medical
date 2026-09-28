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
    "query_consult_graph",
    "search_exam_handbook",
    "search_chroma_knowledge",
    "note_task_remind",
    "note_task_checkin",
    "note_task_medication",
    "note_task_appointment",
    "note_task_report",
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
    rows: list[dict[str, Any]] = []
    for item in list_companion_facts(uid, limit=20):
        kind = str(item.get("kind") or "")
        if kind not in _LIFE_KINDS:
            continue
        name = str(item.get("name") or "").strip()
        fact = str(item.get("fact") or "").strip()
        if q and q not in fact and q not in name and q not in kind:
            continue
        rows.append(
            {
                "kind": kind,
                "name": name,
                "fact": ("已排除：" + name) if item.get("negated") else fact,
                "last_said": item.get("last_said") or "",
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


def _kg_turn_query(query: str) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    turn = _turn()
    name = str(turn.get("handler") or "fallback")
    text = (query or "").strip() or str(turn.get("query") or "")
    mentions = turn.get("mentions") if isinstance(turn.get("mentions"), dict) else {}
    alignments = turn.get("alignments") if isinstance(turn.get("alignments"), list) else []
    return name, text, mentions, alignments


def _kg_dump(tool: str, result: dict[str, Any]) -> str:
    return dump_envelope(
        tool=tool,
        summary=str(result.get("summary") or ""),
        data=result,
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

    _name, text, mentions, _alignments = _kg_turn_query(query)
    extra = []
    for key in ("部位", "疾病", "症状", "检查", "指标"):
        extra.extend(mentions.get(key) or [])
    q = " ".join(part for part in [text, *extra] if part).strip() or text
    return _kg_dump("search_chroma_knowledge", run(query=q, handler=_name))


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
    """判定为报告解读且有 mineru_pdf_path 则 MinerU 拆 PDF；否则用 last_file 切块抽指标。"""
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
    mark_companion_tool(lookup_stated_consult, "memory")
    mark_companion_tool(
        lookup_life_facts,
        "memory",
        majors=("个人记忆问答", "近况陈述"),
    )
    mark_companion_tool(search_long_memory, "memory", majors=("近况陈述",))
    mark_companion_tool(describe_uploaded_image, "file", handlers=("file_image",))
    mark_companion_tool(describe_uploaded_document, "file", handlers=("file_document",))
    mark_companion_tool(explain_uploaded_report, "report", handlers=("report_upload",))
    mark_companion_tool(extract_report_tables, "report", handlers=("report_tables", "report_upload"))
    mark_companion_tool(explain_mentioned_report, "report", handlers=("report_explain",))
    mark_companion_tool(compare_lab_history, "report", handlers=("report_compare",))
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
