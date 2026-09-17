"""进入 C 端 Agent 前的预处理：纠错 → 指代补全 → 意图 → 抽实体对齐。

图谱与长期记忆改由意图工具查询，预处理只给 Agent 上下文。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from algorithm.companion_cypher import list_align_catalogs
from algorithm.coref_rewrite import rewrite_coref_query
from algorithm.intent_classify import (
    CompanionIntentResult,
    classify_companion_intent,
    intent_result_to_dict,
)
from algorithm.life_lexicons import extract_life_mentions
from configs.paths_b2c import b2b_src_imports

logger = logging.getLogger(__name__)

__all__ = [
    "CompanionPreprocess",
    "finish_preprocess_after_intent",
    "format_preprocess_for_agent",
    "run_chat_preprocess",
]

# 这些大类不抽部位/病名、不跑对齐
_SKIP_EXTRACT_MAJOR = frozenset({"闲聊", "风险拦截", "other", "报告指标解读"})
# 寒暄/拦截/未知不搜长期记忆
_SKIP_MEMORY_MAJOR = frozenset({"闲聊", "风险拦截", "other"})


@dataclass
class CompanionPreprocess:
    raw_text: str
    corrected_text: str
    resolved_text: str = ""
    spell: dict[str, Any] = field(default_factory=dict)
    coref: dict[str, Any] = field(default_factory=dict)
    intent: str = "未知意图"
    intent_major: str = "other"
    intent_handler: str = "fallback"
    intent_detail: dict[str, Any] = field(default_factory=dict)
    mentions: dict[str, list[str]] = field(default_factory=dict)
    negated: list[str] = field(default_factory=list)
    negation_spans: list[str] = field(default_factory=list)
    life: dict[str, list[str]] = field(default_factory=dict)
    alignments: list[dict[str, Any]] = field(default_factory=list)
    cypher: dict[str, Any] = field(default_factory=dict)
    kg_result: dict[str, Any] = field(default_factory=dict)
    skip_long_memory: bool = False
    entity_extracted: bool = False
    stage_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_chat_preprocess(
    text: str,
    *,
    user_id: str = "",
    last_preprocess: dict[str, Any] | None = None,
    last_user_text: str = "",
    slot_memory: dict[str, list[str]] | None = None,
    prior_user_texts: list[str] | None = None,
    defer_intent: bool = False,
) -> CompanionPreprocess:
    del user_id
    raw = (text or "").strip()
    if not raw:
        return CompanionPreprocess(raw_text="", corrected_text="", resolved_text="", skip_long_memory=True)

    errors: dict[str, str] = {}
    if raw.lstrip().startswith(("【上传检查报告】", "【上传图片】", "【上传文档】")) or "上传检查报告" in raw[:40]:
        corrected, spell = raw, {"changed": False, "source": "passthrough", "explanation": "上传原文不纠错"}
    else:
        corrected, spell = _spell_correct(raw, errors)
    coref = rewrite_coref_query(
        corrected,
        last_preprocess=last_preprocess,
        last_user_text=last_user_text,
        slot_memory=slot_memory,
        prior_user_texts=prior_user_texts,
    )
    query = coref.text or corrected
    life = extract_life_mentions(query)
    prep = CompanionPreprocess(
        raw_text=raw,
        corrected_text=corrected,
        resolved_text=query,
        spell=spell,
        coref={
            "changed": coref.changed,
            "fillers": coref.fillers,
            "reason": coref.reason,
        },
        life=life,
        stage_errors=errors,
        skip_long_memory=True,
    )
    if defer_intent:
        return prep
    return finish_preprocess_after_intent(prep, classify_companion_intent(query))


def _as_intent_list(classified: Any) -> list[Any]:
    if classified is None:
        return []
    if isinstance(classified, (list, tuple)):
        return [item for item in classified if item is not None]
    return [classified]


def finish_preprocess_after_intent(prep: CompanionPreprocess, classified: Any) -> CompanionPreprocess:
    """意图已由 Agent/LLM 判定后：按大类决定是否抽槽对齐。支持多意图列表。"""
    items = _as_intent_list(classified)
    if not items:
        items = [
            CompanionIntentResult(
                intent="未知意图",
                major="other",
                confidence=0.0,
                source="empty",
                reason="无意图",
                slots=(),
                handler="fallback",
                recommend_action="转人工或兜底",
            )
        ]
    primary = items[0]
    query = prep.resolved_text or prep.corrected_text
    majors = {str(getattr(item, "major", None) or "other") for item in items}
    skip_extract = bool(majors) and all(major in _SKIP_EXTRACT_MAJOR for major in majors)
    skip = bool(majors) and all(major in _SKIP_MEMORY_MAJOR for major in majors)
    errors = dict(prep.stage_errors or {})
    mentions: dict[str, list[str]] = {"部位": [], "疾病": [], "症状": []}
    negated: list[str] = []
    negation_spans: list[str] = []
    alignments: list[dict[str, Any]] = []
    if not skip_extract and query:
        catalogs = list_align_catalogs()
        mentions, negated, negation_spans, alignments = _extract_align(query, catalogs, errors)
    prep.intent = str(getattr(primary, "intent", None) or "未知意图")
    prep.intent_major = str(getattr(primary, "major", None) or "other")
    prep.intent_handler = str(getattr(primary, "handler", None) or "fallback")
    prep.intent_detail = {
        "major": prep.intent_major,
        "handler": prep.intent_handler,
        "slots": list(getattr(primary, "slots", ()) or ()),
        "confidence": float(getattr(primary, "confidence", 0.0) or 0.0),
        "source": str(getattr(primary, "source", "") or ""),
        "reason": str(getattr(primary, "reason", "") or ""),
        "recommend_action": str(getattr(primary, "recommend_action", "") or ""),
        "intents": [
            intent_result_to_dict(item) if isinstance(item, CompanionIntentResult) else {
                "intent": getattr(item, "intent", ""),
                "major": getattr(item, "major", ""),
                "handler": getattr(item, "handler", ""),
                "reason": getattr(item, "reason", ""),
            }
            for item in items
        ],
    }
    prep.mentions = mentions
    prep.negated = negated
    prep.negation_spans = negation_spans
    prep.alignments = alignments
    prep.skip_long_memory = skip
    prep.entity_extracted = not skip_extract
    prep.stage_errors = errors
    return prep


def _spell_correct(text: str, errors: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """C 端自己的 BERT / 纠错 Agent，不绕 B2B（避免 schema/configs 撞名）。"""
    try:
        from runner.Predictor import get_spell_check_bert_predictor

        corrected = (get_spell_check_bert_predictor().predict(text) or "").strip() or text
        return corrected, {
            "changed": corrected != text,
            "source": "bert",
            "explanation": "C端 BERT 逐字纠错",
        }
    except Exception as exc:
        logger.warning("companion BERT spell failed: %s", exc)
        errors["spell_bert"] = str(exc)

    try:
        from agent.spell_check_agent import correct_text

        result = correct_text(text, require_high_confidence=True)
        corrected = (result.corrected_text or "").strip() or text
        return corrected, {
            "changed": corrected != text,
            "source": "agent",
            "confidence": float(result.confidence),
            "explanation": result.explanation or "",
        }
    except Exception as exc:
        logger.warning("companion spell agent failed: %s", exc)
        errors["spell_agent"] = str(exc)
        return text, {"changed": False, "source": "passthrough", "explanation": "纠错不可用，保留原文"}


def _merge_unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in items if str(item).strip()))


def _trusted_part_mentions(part_mentions: list[str], symptom_mentions: list[str]) -> list[str]:
    """UIE 常把「脑袋疼」拆成部位「脑袋」+ 症状「脑袋疼」，部位子串不可信。"""
    trusted: list[str] = []
    for part in part_mentions:
        text = (part or "").strip()
        if not text:
            continue
        if any(text in symptom and text != symptom for symptom in symptom_mentions):
            continue
        trusted.append(text)
    return trusted


def _extract_align(
    text: str,
    catalogs: dict[str, list[dict[str, Any]]],
    errors: dict[str, str],
) -> tuple[dict[str, list[str]], list[str], list[str], list[dict[str, Any]]]:
    """UIE/否定/对齐必须在 B2B 切栈内执行，避免惰性 import 打到 C 端 algorithm。"""
    mentions = {"部位": [], "疾病": [], "症状": []}
    negated: list[str] = []
    negation_spans: list[str] = []
    alignments: list[dict[str, Any]] = []
    try:
        with b2b_src_imports():
            from algorithm.entity_align import entity_align
            from algorithm.negation import drop_negated_mentions, extract_negation_spans
            from algorithm.uie_extract import extract_entity
            from schema.entity_synonyms import collect_synonym_spans

            extracted = extract_entity(text, ["部位", "疾病", "症状"])
            for key, entity_type in (("部位", "body_part"), ("疾病", "disease"), ("症状", "symptom")):
                seeded = [alias for alias, _ in collect_synonym_spans(entity_type, text)]
                mentions[key] = _merge_unique(list(extracted.get(key) or []) + seeded)

            spans = extract_negation_spans(text)
            negation_spans = [getattr(item, "text", str(item)) for item in (spans or [])]
            for key in mentions:
                raw_spans = list(mentions[key])
                kept = drop_negated_mentions(text, raw_spans)
                dropped = [item for item in raw_spans if item not in kept]
                negated.extend(dropped)
                mentions[key] = kept

            mentions["部位"] = _trusted_part_mentions(mentions["部位"], mentions["症状"])
            nlu = {"entity_align": entity_align}
            for mention in mentions["部位"]:
                alignments.append(
                    _align_one(
                        nlu,
                        "body_part",
                        mention,
                        catalogs["body_part"],
                        name_key="part_name",
                        node_label="ConsultBodyPart",
                        utterance=text,
                    )
                )
            kept_diseases: list[str] = []
            for mention in mentions["疾病"]:
                hit = _align_one(
                    nlu,
                    "disease",
                    mention,
                    catalogs["disease"],
                    name_key="disease_name",
                    node_label="ConsultDisease",
                    utterance=text,
                )
                if hit.get("standard_name") and not hit.get("needs_review"):
                    kept_diseases.append(mention)
                    alignments.append(hit)
                else:
                    mentions["症状"] = _merge_unique(list(mentions["症状"]) + [mention])
            mentions["疾病"] = kept_diseases
            for mention in mentions["症状"]:
                alignments.append(
                    _align_one(
                        nlu,
                        "symptom",
                        mention,
                        catalogs["symptom"],
                        name_key="symptom_name",
                        node_label="ConsultPrimarySymptom",
                        utterance=text,
                        keep_span=True,
                    )
                )
    except Exception as exc:
        logger.exception("companion extract/align failed")
        errors["extract_align"] = str(exc)
    return mentions, list(dict.fromkeys(negated)), list(dict.fromkeys(negation_spans)), alignments


def _align_one(
    nlu: dict[str, Any],
    entity_type: str,
    mention: str,
    catalog: list[dict[str, Any]],
    *,
    name_key: str,
    node_label: str,
    utterance: str,
    keep_span: bool = False,
) -> dict[str, Any]:
    hit = nlu["entity_align"](
        entity_type,
        mention,
        catalog,
        name_key=name_key,
        node_label=node_label,
        for_span=True,
        utterance=utterance,
    )
    payload = _hit_dict(hit)
    if keep_span and not payload.get("standard_name"):
        payload["standard_name"] = mention
        payload["method"] = "span"
        payload["needs_review"] = True
    return payload


def _hit_dict(hit: Any) -> dict[str, Any]:
    return {
        "mention": getattr(hit, "mention", ""),
        "standard_name": getattr(hit, "standard_name", None),
        "node_id": getattr(hit, "node_id", None),
        "node_label": getattr(hit, "node_label", None),
        "method": getattr(hit, "method", ""),
        "score": getattr(hit, "score", 0.0),
        "needs_review": getattr(hit, "needs_review", False),
    }


def format_preprocess_for_agent(prep: CompanionPreprocess) -> str:
    """塞进 Human 附注，给 Agent 当结构化观察，不当诊断。"""
    lines = [
        f"纠错后：{prep.corrected_text}",
        f"指代补全：{prep.resolved_text or prep.corrected_text}",
        f"大类：{prep.intent_major}",
        f"细意图：{prep.intent}",
        f"路由：{prep.intent_handler}",
    ]
    extra = (prep.intent_detail or {}).get("intents") or []
    if len(extra) > 1:
        bits = [
            f"{item.get('major')}/{item.get('intent')}({item.get('handler')})"
            for item in extra
            if isinstance(item, dict)
        ]
        if bits:
            lines.append("多意图：" + "；".join(bits))
    if prep.mentions:
        bits = [f"{k}={','.join(v)}" for k, v in prep.mentions.items() if v]
        if bits:
            lines.append("抽出：" + "；".join(bits))
    if prep.negated:
        lines.append("已排除否定：" + "、".join(prep.negated))
    if prep.life:
        bits = [f"{k}={','.join(v)}" for k, v in prep.life.items() if v]
        if bits:
            lines.append("近况：" + "；".join(bits))
    aligned = [
        f"{item.get('mention')}→{item.get('standard_name')}"
        for item in prep.alignments
        if item.get("standard_name")
    ]
    if aligned:
        lines.append("对齐：" + "；".join(aligned))
    lines.append("证据请调用本轮已绑定工具获取，不要编造图谱清单或病史。")
    return "\n".join(lines)
