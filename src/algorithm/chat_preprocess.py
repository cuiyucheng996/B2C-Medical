"""进入 C 端 Agent 前的预处理：纠错 → 指代补全 → 意图 → 抽实体对齐。

图谱与长期记忆改由意图工具查询，预处理只给 Agent 上下文。
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from algorithm.companion_cypher import list_align_catalogs
from algorithm.coref_rewrite import rewrite_coref_query
from algorithm.intent_classify import CompanionIntentResult, classify_companion_intents, intent_result_to_dict
from algorithm.life_lexicons import extract_life_mentions
from configs.paths_b2c import ensure_uvgraph_root

logger = logging.getLogger(__name__)

__all__ = [
    "CompanionPreprocess",
    "finish_preprocess_after_intent",
    "format_preprocess_for_agent",
    "prep_from_dict",
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
    stage_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prep_from_dict(row: dict[str, Any], *, fallback: CompanionPreprocess | None = None) -> CompanionPreprocess:
    base = fallback or CompanionPreprocess(raw_text="", corrected_text="", resolved_text="")
    return CompanionPreprocess(
        raw_text=str(row.get("raw_text") or base.raw_text),
        corrected_text=str(row.get("corrected_text") or base.corrected_text),
        resolved_text=str(row.get("resolved_text") or base.resolved_text),
        spell=dict(row.get("spell") or base.spell or {}),
        coref=dict(row.get("coref") or base.coref or {}),
        intent=str(row.get("intent") or base.intent),
        intent_major=str(row.get("intent_major") or base.intent_major),
        intent_handler=str(row.get("intent_handler") or base.intent_handler),
        intent_detail=dict(row.get("intent_detail") or base.intent_detail or {}),
        mentions=row.get("mentions") if isinstance(row.get("mentions"), dict) else dict(base.mentions or {}),
        negated=list(row.get("negated") or base.negated or []),
        negation_spans=list(row.get("negation_spans") or base.negation_spans or []),
        life=dict(row.get("life") or base.life or {}),
        alignments=list(row.get("alignments") or base.alignments or []),
        cypher=dict(row.get("cypher") or base.cypher or {}),
        kg_result=dict(row.get("kg_result") or base.kg_result or {}),
        skip_long_memory=bool(row.get("skip_long_memory", base.skip_long_memory)),
        entity_extracted=bool(row.get("entity_extracted", base.entity_extracted)),
        stage_errors=dict(row.get("stage_errors") or base.stage_errors or {}),
        stage_ms=dict(row.get("stage_ms") or base.stage_ms or {}),
    )


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
            "source": coref.source,
        },
        life=life,
        stage_errors=errors,
        skip_long_memory=True,
    )
    if defer_intent:
        return prep
    return finish_preprocess_after_intent(prep, classify_companion_intents(query))


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
    stage_ms = dict(prep.stage_ms or {})
    if not skip_extract and query:
        t_cat = time.perf_counter()
        catalogs = list_align_catalogs()
        stage_ms["catalog_ms"] = int((time.perf_counter() - t_cat) * 1000)
        t_align = time.perf_counter()
        mentions, negated, negation_spans, alignments = _extract_align(query, catalogs, errors)
        stage_ms["extract_align_ms"] = int((time.perf_counter() - t_align) * 1000)
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
    prep.stage_ms = stage_ms
    return prep


def _spell_correct(text: str, errors: dict[str, str]) -> tuple[str, dict[str, Any]]:
    """先 DeepSeek 纠错；调用失败再降级 BERT。已写对的医学词不改。"""
    try:
        corrected = _deepseek_spell(text)
        corrected = _keep_known_terms(text, corrected)
        return corrected, {
            "changed": corrected != text,
            "source": "deepseek",
            "explanation": "DeepSeek 纠错；已有化验名和图谱标准名不改",
        }
    except Exception as exc:
        logger.warning("companion DeepSeek spell failed: %s", exc)
        errors["spell_deepseek"] = str(exc)

    try:
        from runner.Predictor import get_spell_check_bert_predictor

        corrected = (get_spell_check_bert_predictor().predict(text) or "").strip() or text
        corrected = _keep_known_terms(text, corrected)
        return corrected, {
            "changed": corrected != text,
            "source": "bert",
            "explanation": "DeepSeek 不可用，降级 BERT；已有化验名和图谱标准名不改",
        }
    except Exception as exc:
        logger.warning("companion BERT spell failed: %s", exc)
        errors["spell_bert"] = str(exc)
        return text, {"changed": False, "source": "passthrough", "explanation": "纠错不可用，保留原文"}


def _deepseek_spell(text: str) -> str:
    ensure_uvgraph_root()
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.llm.chat import get_chat_model

    model = get_chat_model()
    if model is None:
        raise RuntimeError("未配置聊天模型")
    resp = model.invoke(
        [
            SystemMessage(
                content=(
                    "你只做中文错别字纠正。医学化验名、病名、症状名如果原文已经正确就不要改。"
                    "只输出纠正后的整句，不要解释。"
                )
            ),
            HumanMessage(content=text),
        ]
    )
    out = str(getattr(resp, "content", "") or "").strip()
    if not out:
        raise RuntimeError("DeepSeek 纠错空结果")
    return out


_LAB_TERM_RE = re.compile(
    r"转氨酶|谷丙转氨酶|谷草转氨酶|肌酐|尿素氮|尿酸|血糖|胆固醇|甘油三酯|"
    r"血红蛋白|白细胞|红细胞|血小板|血常规|肝功能|肾功能|甲功"
)


def _keep_known_terms(original: str, corrected: str) -> str:
    """原文里已经写对的化验名、部位、疾病、症状，纠错不得改掉。"""
    src = original or ""
    dst = corrected or ""
    if not src or src == dst:
        return dst or src
    terms = _known_terms_in(src)
    if not terms:
        return dst
    if len(src) != len(dst):
        if any(term not in dst for term in terms):
            return src
        return dst
    chars = list(dst)
    for term in sorted(terms, key=len, reverse=True):
        start = 0
        while True:
            idx = src.find(term, start)
            if idx < 0:
                break
            chars[idx : idx + len(term)] = list(term)
            start = idx + len(term)
    return "".join(chars)


def _known_terms_in(text: str) -> list[str]:
    found = [item.group(0) for item in _LAB_TERM_RE.finditer(text)]
    try:
        catalogs = list_align_catalogs()
    except Exception:
        logger.debug("spell protect catalogs unavailable", exc_info=True)
        catalogs = {}
    keys = {
        "body_part": "part_name",
        "disease": "disease_name",
        "symptom": "symptom_name",
    }
    for bucket, field in keys.items():
        for row in catalogs.get(bucket) or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get(field) or "").strip()
            if len(name) >= 2 and name in text:
                found.append(name)
    return list(dict.fromkeys(found))


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
    """UIE/否定/对齐走 B2B 桥接缓存，不切换 C 端 algorithm 包。"""
    mentions = {"部位": [], "疾病": [], "症状": []}
    negated: list[str] = []
    negation_spans: list[str] = []
    alignments: list[dict[str, Any]] = []
    try:
        from algorithm.entity_align_c import entity_align
        from b2c_b2b_bridge import b2b_entity_synonyms, b2b_negation, b2b_uie_extract
        drop_negated_mentions = b2b_negation().drop_negated_mentions
        extract_negation_spans = b2b_negation().extract_negation_spans
        extract_entity = b2b_uie_extract().extract_entity
        collect_synonym_spans = b2b_entity_synonyms().collect_synonym_spans

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
