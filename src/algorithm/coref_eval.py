"""评测台：指代补金标。只调 rewrite_coref_query，不跑陪聊 LLM。"""

from __future__ import annotations

from typing import Any

from algorithm.coref_rewrite import rewrite_coref_query

__all__ = ["COREF_CASES", "run_coref_eval", "run_coref_one"]

COREF_CASES: list[dict[str, Any]] = [
    {
        "id": "disease_cue",
        "title": "这个病 → 病名",
        "query": "这个病严重吗",
        "prior": "",
        "slot_memory": {"疾病": ["胃炎"]},
        "expect_changed": True,
        "expect_source": "slot",
        "contains": ["胃炎"],
        "not_contains": ["这个病"],
    },
    {
        "id": "metric_cue",
        "title": "这个指标 → ALT",
        "query": "这个指标偏高危险吗",
        "prior": "",
        "slot_memory": {"指标": ["ALT"]},
        "expect_changed": True,
        "expect_source": "slot",
        "contains": ["ALT"],
        "not_contains": ["这个指标"],
    },
    {
        "id": "exam_cue",
        "title": "这个检查 → 血常规",
        "query": "这个检查是查什么的",
        "prior": "",
        "slot_memory": {"检查": ["血常规"]},
        "expect_changed": True,
        "expect_source": "slot",
        "contains": ["血常规"],
        "not_contains": ["这个检查"],
    },
    {
        "id": "follow_danger",
        "title": "那危险吗 ← 上一句 ALT",
        "query": "那危险吗",
        "prior": "ALT 偏高是什么意思",
        "slot_memory": {},
        "expect_changed": True,
        "expect_source": "slot",
        "contains": ["ALT"],
        "not_contains": [],
    },
    {
        "id": "follow_five",
        "title": "那不知道这五项 ← 模型补「五事」",
        "query": "那不知道这五项会怎样？",
        "prior": "孙子说的「五事」分别是什么？",
        "slot_memory": {},
        "expect_changed": True,
        "expect_source": "llm",
        "contains": ["五事"],
        "not_contains": [],
    },
    {
        "id": "it_metric",
        "title": "它 → 指标名",
        "query": "它正常吗",
        "prior": "",
        "slot_memory": {"指标": ["空腹血糖"]},
        "expect_changed": True,
        "expect_source": "slot",
        "contains": ["空腹血糖"],
        "not_contains": [],
    },
    {
        "id": "new_topic",
        "title": "新话题不接 ALT",
        "query": "那我最近失眠怎么办",
        "prior": "ALT 偏高是什么意思",
        "slot_memory": {"指标": ["ALT"]},
        "expect_changed": False,
        "expect_source": "none",
        "contains": [],
        "not_contains": ["ALT："],
    },
    {
        "id": "no_coref",
        "title": "完整问句不改",
        "query": "头部常见有哪些病",
        "prior": "胃炎有哪些症状",
        "slot_memory": {"疾病": ["胃炎"]},
        "expect_changed": False,
        "expect_source": "none",
        "contains": [],
        "not_contains": [],
    },
]


def _score_case(case: dict[str, Any]) -> dict[str, Any]:
    prior = str(case.get("prior") or "").strip()
    out = rewrite_coref_query(
        str(case.get("query") or ""),
        last_user_text=prior,
        slot_memory=case.get("slot_memory") if isinstance(case.get("slot_memory"), dict) else {},
        prior_user_texts=[prior] if prior else None,
    )
    want_changed = bool(case.get("expect_changed"))
    ok_changed = out.changed is want_changed
    missing = [item for item in (case.get("contains") or []) if item not in out.text]
    leaked = [item for item in (case.get("not_contains") or []) if item in out.text]
    want_source = str(case.get("expect_source") or "")
    ok_source = (not want_source) or out.source == want_source
    ok = ok_changed and ok_source and not missing and not leaked
    fail: list[str] = []
    if not ok_changed:
        fail.append("changed")
    if not ok_source:
        fail.append("source:" + (out.source or "-"))
    if missing:
        fail.append("missing:" + ",".join(missing))
    if leaked:
        fail.append("leaked:" + ",".join(leaked))
    return {
        "id": case["id"],
        "title": case.get("title") or case["id"],
        "ok": ok,
        "query": case.get("query") or "",
        "prior": prior or "（无）",
        "slots": case.get("slot_memory") or {},
        "expect_changed": want_changed,
        "expect_source": want_source or "-",
        "got_changed": out.changed,
        "got_source": out.source or "-",
        "got_text": out.text,
        "got_reason": out.reason,
        "fail": fail,
    }


def run_coref_eval() -> dict[str, Any]:
    rows = [_score_case(case) for case in COREF_CASES]
    passed = sum(1 for row in rows if row.get("ok"))
    return {
        "ok": passed == len(rows),
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }


def run_coref_one(
    *,
    query: str,
    prior: str = "",
    slot_memory: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    text = (query or "").strip()
    if not text:
        return {"ok": False, "reason": "empty"}
    hist = (prior or "").strip()
    out = rewrite_coref_query(
        text,
        last_user_text=hist,
        slot_memory=slot_memory or {},
        prior_user_texts=[hist] if hist else None,
    )
    return {
        "ok": True,
        "query": text,
        "prior": hist,
        "slots": slot_memory or {},
        "got_changed": out.changed,
        "got_source": out.source or "-",
        "got_text": out.text,
        "got_reason": out.reason,
        "fillers": out.fillers,
    }
