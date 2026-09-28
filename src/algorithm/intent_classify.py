"""意图：第三方模型同时判大类+细意图；失败再降级 BERT 大类 + 规则细类。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from algorithm.intent_catalog import INTENT_BY_NAME, INTENT_ROWS, intent_prompt_lines, major_of
from algorithm.intent_rule import match_all_intents, match_intent
from configs.paths_b2c import B2C_ROOT, ensure_uvgraph_root
from prompt import load_prompt

logger = logging.getLogger(__name__)

__all__ = [
    "CompanionIntentResult",
    "classify_companion_intent",
    "classify_companion_intents",
    "classify_companion_intents_bert",
    "classify_companion_intent_llm",
    "resolve_companion_intent_items",
    "classify_fine_for_major",
    "classify_intent_fallback",
    "classify_intent_fallback_multi",
    "intent_result_to_dict",
    "intent_score_threshold",
    "major_score_threshold",
    "major_scores_to_rows",
    "major_bert_available",
    "normalize_intent_results",
    "sort_major_scores",
]
_MAX_INTENTS_DEFAULT = 5

_MAJORS = (
    "公共知识问答",
    "具体任务",
    "个人记忆问答",
    "报告指标解读",
    "风险拦截",
    "近况陈述",
    "闲聊",
    "other",
)
_JSON_RE = re.compile(r"\{[\s\S]*\}")
_BERT_CUTOFF = 0.7


@dataclass(frozen=True)
class CompanionIntentResult:
    intent: str
    major: str
    confidence: float
    source: str
    reason: str
    slots: tuple[str, ...]
    handler: str
    recommend_action: str = "正常处理"
    # 全大类打分：8 个大类各一个 0~1 分，供测试台与阈值判定共用
    major_scores: dict[str, float] = field(default_factory=dict)


def _major_model_dir() -> Path:
    return B2C_ROOT / "checkpoint" / "intent_major" / "best_model"


def major_bert_available() -> bool:
    """训练后大意图 BERT 权重是否就绪（不加载模型）。"""
    model_dir = _major_model_dir()
    mapping_path = model_dir.parent / "label_mapping.json"
    return model_dir.is_dir() and mapping_path.is_file()


@lru_cache(maxsize=1)
def _major_bert_bundle() -> tuple[Any, Any, dict[str, str]] | None:
    model_dir = _major_model_dir()
    mapping_path = model_dir.parent / "label_mapping.json"
    if not model_dir.is_dir() or not mapping_path.is_file():
        return None
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with mapping_path.open(encoding="utf-8") as f:
        mapping = {str(k): str(v) for k, v in json.load(f).items()}
    return model, tokenizer, mapping


def _empty_major_scores() -> dict[str, float]:
    """全大类打分骨架：每个大类先记 0 分。"""
    return {name: 0.0 for name in _MAJORS}


def _clamp_major_scores(raw: Any) -> dict[str, float]:
    """把模型/BERT 给的分数补齐成全部大类，缺的与大类外的都记 0。"""
    scores = _empty_major_scores()
    if not isinstance(raw, dict):
        return scores
    for key, value in raw.items():
        name = str(key or "").strip()
        if name not in scores:
            continue
        try:
            scores[name] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return scores


def sort_major_scores(scores: dict[str, float] | None) -> list[tuple[str, float]]:
    """全大类打分按分数降序，供阈值判定与测试台展示共用。"""
    if not scores:
        return []
    return sorted(scores.items(), key=lambda row: (-float(row[1]), row[0]))


def major_scores_to_rows(scores: dict[str, float] | None) -> list[dict[str, Any]]:
    """全大类打分 → 展示行（降序），前端与接口统一用这个形状。"""
    return [
        {"major": name, "confidence": round(float(value), 4)}
        for name, value in sort_major_scores(scores)
    ]


def _pack(
    *,
    major: str,
    intent: str,
    confidence: float,
    source: str,
    reason: str,
    major_scores: dict[str, float] | None = None,
) -> CompanionIntentResult:
    scores = _clamp_major_scores(major_scores) if major_scores else {}
    if major not in _MAJORS or major == "other" or intent in ("未知意图", "other", ""):
        return CompanionIntentResult(
            intent="未知意图",
            major="other",
            confidence=confidence,
            source=source,
            reason=reason,
            slots=(),
            handler="fallback",
            recommend_action="转人工或兜底",
            major_scores=scores,
        )
    row = INTENT_BY_NAME.get(intent) or {}
    handler = str(row.get("handler") or "fallback")
    if intent == "未细分":
        handler = "fallback"
    return CompanionIntentResult(
        intent=intent,
        major=major,
        confidence=confidence,
        source=source,
        reason=reason,
        slots=tuple(row.get("slots") or ()),
        handler=handler,
        recommend_action="正常处理",
        major_scores=scores,
    )


def intent_result_to_dict(item: CompanionIntentResult) -> dict[str, Any]:
    return {
        "intent": item.intent,
        "major": item.major,
        "handler": item.handler,
        "slots": list(item.slots),
        "confidence": item.confidence,
        "source": item.source,
        "reason": item.reason,
        "recommend_action": item.recommend_action,
    }


def intent_score_threshold() -> float:
    _, thresh = _intent_multi_settings()
    return thresh


def major_score_threshold() -> float:
    """全大类打分进候选的阈值；未单独配置时沿用多意图阈值。"""
    _, thresh = _intent_multi_settings()
    try:
        ensure_uvgraph_root()
        from src.configs import get_yaml_config

        block = ((get_yaml_config() or {}).get("b2b_api") or {}).get("intent_classify") or {}
        if block.get("major_score_threshold") is not None:
            return max(0.0, min(1.0, float(block["major_score_threshold"])))
    except Exception:
        logger.debug("major score threshold fallback", exc_info=True)
    return thresh


def _majors_above_threshold(scores: dict[str, float] | None) -> list[tuple[str, float]]:
    """全大类打分里达到阈值的大类（降序），闲聊/other 不进候选。"""
    cutoff = major_score_threshold()
    return [
        (name, value)
        for name, value in sort_major_scores(scores)
        if value >= cutoff and name not in {"other", "闲聊"}
    ]


def _intent_multi_settings() -> tuple[int, float]:
    max_n = _MAX_INTENTS_DEFAULT
    thresh = 0.55
    try:
        ensure_uvgraph_root()
        from src.configs import get_yaml_config

        block = ((get_yaml_config() or {}).get("b2b_api") or {}).get("intent_classify") or {}
        if block.get("multi_intent_threshold") is not None:
            thresh = float(block["multi_intent_threshold"])
        elif block.get("score_threshold") is not None:
            thresh = float(block["score_threshold"])
        if block.get("max_intents") is not None:
            max_n = int(block["max_intents"])
    except Exception:
        logger.debug("intent multi settings fallback", exc_info=True)
    return max(1, max_n), max(0.0, min(1.0, thresh))


def normalize_intent_results(
    items: list[CompanionIntentResult],
    *,
    min_confidence: float | None = None,
) -> list[CompanionIntentResult]:
    """按置信度阈值过滤、去重、去掉被实义意图挤掉的闲聊/other。"""
    max_n, default_thresh = _intent_multi_settings()
    cutoff = default_thresh if min_confidence is None else float(min_confidence)
    packed: list[CompanionIntentResult] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if item is None:
            continue
        key = (item.major, item.handler)
        if key in seen:
            continue
        seen.add(key)
        packed.append(item)
    packed.sort(key=lambda row: (-float(row.confidence or 0), row.major, row.intent))
    above = [item for item in packed if float(item.confidence or 0) >= cutoff]
    if not above and packed:
        above = [packed[0]]
    real = [item for item in above if item.major not in {"闲聊", "other"}]
    chosen = real or above or packed
    if not chosen:
        chosen = [_pack(major="other", intent="未知意图", confidence=0.0, source="empty", reason="无意图")]
    return chosen[:max_n]


def _intent_catalog_prompt() -> str:
    _, thresh = _intent_multi_settings()
    return load_prompt("intent_catalog").format(
        majors="、".join(_MAJORS),
        intent_lines=intent_prompt_lines(),
        score_threshold=thresh,
    )


def _from_llm_fields(
    major: str,
    intent: str,
    reason: str,
    *,
    confidence: float = 0.8,
) -> CompanionIntentResult:
    major = (major or "").strip()
    intent = (intent or "").strip()
    if intent in INTENT_BY_NAME:
        major = major_of(intent)
    elif intent != "未细分":
        intent = "未细分"
    if major not in _MAJORS:
        major = "other"
    score = max(0.0, min(1.0, float(confidence)))
    return _pack(
        major=major,
        intent=intent if major != "other" else "未知意图",
        confidence=score,
        source="llm",
        reason=(reason or "").strip() or "llm",
    )


def _items_from_major_scores(scores: dict[str, float] | None) -> list[CompanionIntentResult]:
    """细意图没给出来时，用达阈值的全大类打分兜出「未细分」。"""
    return [
        _pack(
            major=name,
            intent="未细分",
            confidence=value,
            source="llm-major-score",
            reason=f"细意图缺失，全大类打分 {value}",
            major_scores=scores,
        )
        for name, value in _majors_above_threshold(scores)
    ]


def _reconcile_intents_with_major_scores(
    packed: list[CompanionIntentResult],
    scores: dict[str, float],
    text: str,
    model: Any = None,
) -> list[CompanionIntentResult]:
    """业务规则：大意图分 ≥ major_score_threshold 即视为存在，补全细意图并进入并行任务。"""
    clamped = _clamp_major_scores(scores)
    majors_required = [name for name, _value in _majors_above_threshold(clamped)]
    fine_cutoff, _ = _intent_multi_settings()
    by_major: dict[str, CompanionIntentResult] = {}

    for item in packed:
        major = str(item.major or "").strip()
        if major in {"闲聊", "other", ""}:
            continue
        major_conf = float(clamped.get(major, 0.0) or 0.0)
        if float(item.confidence or 0) >= fine_cutoff or major_conf >= major_score_threshold():
            prev = by_major.get(major)
            if prev is None or float(item.confidence or 0) >= float(prev.confidence or 0):
                by_major[major] = replace(item, major_scores=clamped)

    for name, value in _majors_above_threshold(clamped):
        if name in by_major:
            continue
        row = classify_fine_for_major(text, major=name, model=model, llm_seconds=5)
        by_major[name] = replace(
            row,
            confidence=max(float(row.confidence or 0), float(value)),
            major_scores=clamped,
            source=f"{row.source}+major-threshold",
            reason=f"{row.reason}；大类打分 {value}≥{major_score_threshold()}",
        )

    if by_major:
        ordered = sorted(
            by_major.values(),
            key=lambda row: (-float(clamped.get(row.major, row.confidence) or 0), row.major),
        )
        return ordered
    if packed:
        return [replace(item, major_scores=clamped) if not item.major_scores else item for item in packed]
    return _items_from_major_scores(clamped)


def _attach_major_scores(
    items: list[CompanionIntentResult],
    scores: dict[str, float] | None,
) -> list[CompanionIntentResult]:
    """给还没有全大类打分的结果补分；全 0 视为没打到，不改动。"""
    packed = _clamp_major_scores(scores)
    if not any(packed.values()):
        return items
    return [
        item if item.major_scores else replace(item, major_scores=dict(packed))
        for item in items
    ]


def _ensure_major_scores(
    items: list[CompanionIntentResult],
    text: str,
) -> list[CompanionIntentResult]:
    """任一条缺打分时才跑一次 BERT 全类别 softmax 兜底。"""
    if not items or all(item.major_scores for item in items):
        return items
    return _attach_major_scores(items, _predict_major_scores_bert(text))


def classify_companion_intent_llm(
    model: Any,
    query: str,
    *,
    recent_dialogue: str = "",
) -> list[CompanionIntentResult] | None:
    """第三方模型同时判大类和细意图；失败返回 None，由调用方走 BERT+规则。"""
    text = (query or "").strip()
    if not text or model is None:
        return None
    ensure_uvgraph_root()
    from pydantic import BaseModel, Field

    from src.langchain.agent.agent_stream import _invoke_structured_output

    class IntentItem(BaseModel):
        major: str = Field(description="大类名称，可与 intent 一致时由 intent 反推")
        intent: str = Field(description="细意图名称或未细分")
        confidence: float = Field(description="0~1，该意图与本轮用户话的相关度")
        reason: str = Field(description="短理由")

    class MajorScore(BaseModel):
        major: str = Field(description="大类名称，必须覆盖清单里的每一个大类")
        confidence: float = Field(description="0~1，用户本轮话与该大类的相关度；不相关也要给低分")

    class IntentChoices(BaseModel):
        major_scores: list[MajorScore] = Field(
            description="全部大类各一条、一条都不能少；按相关度从高到低"
        )
        items: list[IntentItem] = Field(
            description="所有 confidence 达阈值的细意图，可多条；按相关度从高到低"
        )

    human = f"【近期对话】\n{recent_dialogue or '（无）'}\n\n【本轮用户】\n{text}"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from algorithm.llm_trace import STAGE_INTENT_MAIN, record_llm_call, serialize_langchain_messages

        msgs = [SystemMessage(content=_intent_catalog_prompt()), HumanMessage(content=human)]
        choice = _invoke_structured_output(model, IntentChoices, msgs)
        try:
            resp_blob = choice.model_dump_json(indent=2) if hasattr(choice, "model_dump_json") else str(choice)
        except Exception:
            resp_blob = str(choice)
        record_llm_call(
            STAGE_INTENT_MAIN,
            serialize_langchain_messages(msgs),
            response=resp_blob,
            label="classify_companion_intent_llm",
        )
    except Exception:
        logger.warning("llm intent classify failed", exc_info=True)
        return None
    raw_items = list(getattr(choice, "items", None) or [])
    scores = _clamp_major_scores(
        {
            str(row.major or "").strip(): float(getattr(row, "confidence", 0.0) or 0.0)
            for row in (getattr(choice, "major_scores", None) or [])
        }
    )
    packed = [
        _prefer_symptom_care(
            _fill_fine(
                _from_llm_fields(
                    str(item.major or ""),
                    str(item.intent or ""),
                    str(item.reason or ""),
                    confidence=float(getattr(item, "confidence", 0.8) or 0.8),
                ),
                query,
            ),
            query,
        )
        for item in raw_items
    ]
    packed = _reconcile_intents_with_major_scores(packed, scores, text, model)
    if not packed:
        return None
    packed = normalize_intent_results(packed)
    return [replace(row, major_scores=scores) for row in packed]


def _fine_intent_llm_timed(
    llm_model: Any,
    raw: str,
    locked: str,
    names: set[str],
    *,
    seconds: float = 12,
) -> tuple[str, str] | None:
    """细意图只打一轮普通 JSON，限时返回；超时不当成成功。"""

    def _call() -> tuple[str, str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        lines = intent_prompt_lines(major=locked) or "\n".join(f"- {name}" for name in sorted(names) if name)
        system = (
            f"大类已锁定为「{locked}」，不要改大类。只从下列细意图选一条：\n{lines}\n"
            "问怎么止血、怎么护理、症状怎么处理、注意什么：选「查询某症状的生活注意事项」，不要填未细分。\n"
            "只输出 JSON：{{\"intent\":\"清单中的名字或未细分\",\"reason\":\"短理由\"}}"
        )
        invoke_model = llm_model
        if hasattr(llm_model, "model_copy"):
            try:
                invoke_model = llm_model.model_copy(update={"timeout": float(seconds)})
            except Exception:
                invoke_model = llm_model
        from algorithm.llm_trace import STAGE_INTENT_FINE, record_llm_call, serialize_langchain_messages

        msgs = [SystemMessage(content=system), HumanMessage(content=raw[:400])]
        resp = invoke_model.invoke(msgs)
        content = str(getattr(resp, "content", "") or "")
        record_llm_call(
            STAGE_INTENT_FINE,
            serialize_langchain_messages(msgs),
            response=content,
            label=f"classify_fine_for_major · {locked}",
        )
        match = _JSON_RE.search(content)
        if not match:
            raise ValueError("fine intent json missing")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("fine intent json invalid")
        return str(data.get("intent") or "").strip(), str(data.get("reason") or "llm").strip()

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    from algorithm.llm_trace import run_in_llm_trace_context

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(lambda: run_in_llm_trace_context(_call)).result(timeout=seconds)
    except FuturesTimeout:
        logger.warning("fine intent llm timeout after %ss", seconds)
        return None
    except Exception:
        logger.warning("fine intent llm failed major=%s", locked, exc_info=True)
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


_CARE_INTENT = "查询某症状的生活注意事项"


def _prefer_symptom_care(item: CompanionIntentResult, text: str) -> CompanionIntentResult:
    """止血/护理常识走手册，不让模型标成未细分或急症拦截。"""
    if match_intent(text, major="公共知识问答") != _CARE_INTENT:
        return item
    if item.intent == _CARE_INTENT and item.major == "公共知识问答":
        return item
    return _pack(
        major="公共知识问答",
        intent=_CARE_INTENT,
        confidence=max(item.confidence, 0.85),
        source=f"{item.source}+rule",
        reason=(item.reason or "") + "；护理/止血走生活注意事项",
        major_scores=item.major_scores,
    )


def classify_fine_for_major(
    text: str,
    *,
    major: str,
    model: Any = None,
    llm_seconds: float = 5,
) -> CompanionIntentResult:
    """大类已锁定时只让第三方模型选细意图；5 秒内必须给出，超时改规则。"""
    raw = (text or "").strip()
    locked = (major or "").strip()
    if not raw:
        return _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本")
    if locked not in _MAJORS or locked == "other":
        return classify_intent_fallback(raw)
    names = {str(row.get("intent") or "") for row in INTENT_ROWS if str(row.get("major") or "") == locked}
    llm_model = model if model is not None else _default_chat_model()
    if llm_model is not None:
        picked = _fine_intent_llm_timed(llm_model, raw, locked, names, seconds=llm_seconds)
        if picked:
            intent_name, reason = picked
            if intent_name in names:
                packed = _pack(major=locked, intent=intent_name, confidence=0.65, source="llm", reason=reason)
            else:
                packed = _pack(major=locked, intent="未细分", confidence=0.6, source="llm", reason=reason)
            return _prefer_symptom_care(_fill_fine(packed, raw), raw)
    fine = match_intent(raw, major=locked)
    if fine:
        return _prefer_symptom_care(
            _pack(major=locked, intent=fine, confidence=0.7, source="rule", reason="细意图规则降级"),
            raw,
        )
    return _prefer_symptom_care(
        _pack(major=locked, intent="未细分", confidence=0.4, source="rule", reason="细意图未命中"),
        raw,
    )


def _predict_major_scores_bert(text: str) -> dict[str, float] | None:
    """BERT 走全类别 softmax，对全部大类都打分（模型没覆盖的类别留 0）。"""
    bundle = _major_bert_bundle()
    if bundle is None:
        return None
    import torch

    model, tokenizer, mapping = bundle
    device = next(model.parameters()).device
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    scores = _empty_major_scores()
    for idx, prob in enumerate(probs.tolist()):
        label = mapping.get(str(idx), "other")
        if label == "chitchat":
            label = "other"
        if label not in scores:
            continue
        scores[label] = max(scores[label], round(float(prob), 4))
    return scores


def _top_major_bert(scores: dict[str, float] | None) -> tuple[str, float] | None:
    """BERT 全大类打分里取最高的一条。"""
    ranked = sort_major_scores(scores)
    if not ranked:
        return None
    label, confidence = ranked[0]
    if label not in _MAJORS:
        label = "other"
    return label, round(float(confidence), 4)


def _predict_major_bert(text: str) -> tuple[str, float] | None:
    return _top_major_bert(_predict_major_scores_bert(text))


def _qwen_major(text: str) -> tuple[str, str] | None:
    ensure_uvgraph_root()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        from src.configs import get_app_settings, get_yaml_config
    except Exception:
        logger.warning("qwen major fallback imports failed", exc_info=True)
        return None
    settings = get_app_settings()
    api_key = (getattr(settings, "dashscope_api_key", None) or "").strip()
    if not api_key:
        logger.warning("qwen major fallback skipped: DASHSCOPE_API_KEY empty")
        return None
    yaml_cfg = get_yaml_config() or {}
    cfg = ((yaml_cfg.get("b2b_api") or {}).get("qwen_fallback") or {})
    if cfg.get("enabled") is False:
        return None
    model = str(cfg.get("model") or "qwen-turbo")
    base_url = str(cfg.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    try:
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0, max_tokens=200)
        resp = llm.invoke(
            [SystemMessage(content=load_prompt("intent_major_qwen")), HumanMessage(content=f"用户原话：{text[:400]}")]
        )
        content = str(getattr(resp, "content", "") or "")
    except Exception:
        logger.warning("qwen-turbo major fallback call failed", exc_info=True)
        return None
    match = _JSON_RE.search(content)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    major = str(data.get("major") or "other").strip()
    if major not in _MAJORS:
        major = "other"
    reason = str(data.get("reason") or "qwen-turbo").strip()
    return major, reason


def _rewrite_synonyms(text: str) -> str:
    try:
        from b2c_b2b_bridge import b2b_entity_synonyms

        mapped = b2b_entity_synonyms().rewrite_synonyms_in_text(text)
    except Exception:
        logger.warning("intent synonym rewrite failed", exc_info=True)
        return text
    return mapped or text


def _default_chat_model() -> Any:
    try:
        ensure_uvgraph_root()
        from src.llm.chat import get_chat_model

        return get_chat_model()
    except Exception:
        logger.warning("intent llm model unavailable", exc_info=True)
        return None


def _fill_fine(item: CompanionIntentResult, text: str) -> CompanionIntentResult:
    if item.major in {"other", "闲聊"}:
        return item
    if item.intent not in ("未细分", "未知意图", ""):
        return item
    fine = match_intent(text, major=item.major)
    if not fine and item.major == "具体任务" and any(k in text for k in ("提醒", "赶飞机")):
        if not any(k in text for k in ("吃药", "服药", "打卡", "复诊", "挂号", "取报告")):
            fine = "提醒用户某日复查或随访"
    if not fine:
        return item
    return _pack(
        major=item.major,
        intent=fine,
        confidence=item.confidence,
        source=f"{item.source}+rule",
        reason=(item.reason or "") + "；细意图规则补全",
        major_scores=item.major_scores,
    )


def classify_intent_fallback_multi(text: str) -> list[CompanionIntentResult]:
    """规则 + BERT 大类：关键词可命中多条细意图。"""
    raw = (text or "").strip()
    if not raw:
        return [
            _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本"),
        ]
    mapped = _rewrite_synonyms(raw)
    query = mapped or raw
    hits = match_all_intents(raw) or match_all_intents(query)
    if hits:
        packed = [
            _prefer_symptom_care(
                _pack(
                    major=major_of(name),
                    intent=name,
                    confidence=0.72,
                    source="rule",
                    reason="多意图关键词命中",
                ),
                raw,
            )
            for name in hits
            if name in INTENT_BY_NAME
        ]
        if packed:
            return normalize_intent_results(packed)
    single = classify_intent_fallback(raw)
    return normalize_intent_results([single])


def classify_intent_fallback(text: str) -> CompanionIntentResult:
    raw = (text or "").strip()
    if not raw:
        return _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本")

    mapped = _rewrite_synonyms(raw)
    query = mapped or raw

    major = "other"
    source = "none"
    confidence = 0.0
    reason = ""

    bert_scores = _predict_major_scores_bert(query)
    bert = _top_major_bert(bert_scores)
    if bert:
        major, confidence = bert
        source = "bert"
        reason = "同义词映射后大意图 BERT"
        unknown = major == "other" or confidence < _BERT_CUTOFF
        if unknown:
            qwen = _qwen_major(query)
            if qwen:
                major, reason = qwen
                source = "qwen"
                confidence = max(confidence, 0.55)
                reason = f"BERT 未知/低置信({bert[1]}) → qwen：{reason}"
            else:
                major = "other"
                reason = f"BERT 未知/低置信({confidence}) 且 qwen 不可用"
    else:
        qwen = _qwen_major(query)
        if qwen:
            major, reason = qwen
            source = "qwen"
            confidence = 0.55
            reason = "无大意图 BERT，qwen：" + reason
        else:
            return _pack(
                major="other",
                intent="未知意图",
                confidence=0.0,
                source="none",
                reason="无大意图模型且 qwen 不可用",
                major_scores=bert_scores,
            )

    if major == "other":
        return _pack(
            major="other",
            intent="未知意图",
            confidence=confidence,
            source=source,
            reason=reason,
            major_scores=bert_scores,
        )

    fine = match_intent(raw, major=major) or match_intent(query, major=major)
    if fine:
        packed = _pack(
            major=major,
            intent=fine,
            confidence=confidence,
            source=source,
            reason=reason + "；细规则",
            major_scores=bert_scores,
        )
    else:
        packed = _pack(
            major=major,
            intent="未细分",
            confidence=confidence,
            source=source,
            reason=reason + "；细意图未命中",
            major_scores=bert_scores,
        )
    return _prefer_symptom_care(packed, raw)


def classify_companion_intents_bert(text: str) -> list[CompanionIntentResult]:
    """评测专用：大意图用训练 BERT 全类别 softmax，细意图规则补全，不走 LLM/qwen。"""
    raw = (text or "").strip()
    if not raw:
        return [
            _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本"),
        ]
    mapped = _rewrite_synonyms(raw)
    query = mapped or raw
    scores = _predict_major_scores_bert(query)
    if scores is None:
        return [
            _pack(
                major="other",
                intent="未知意图",
                confidence=0.0,
                source="none",
                reason="无大意图 BERT 权重（checkpoint/intent_major/best_model）",
            ),
        ]
    packed: list[CompanionIntentResult] = []
    hits = match_all_intents(raw) or match_all_intents(query)
    if hits:
        for name in hits:
            if name not in INTENT_BY_NAME:
                continue
            packed.append(
                _prefer_symptom_care(
                    _pack(
                        major=major_of(name),
                        intent=name,
                        confidence=0.72,
                        source="rule",
                        reason="BERT 评测：关键词命中细意图",
                        major_scores=scores,
                    ),
                    raw,
                )
            )
    packed = _reconcile_intents_with_major_scores(packed, scores, raw, model=None)
    packed = normalize_intent_results(packed)
    return [_prefer_symptom_care(item, raw) for item in packed]


def classify_companion_intents(
    text: str,
    model: Any = None,
    *,
    recent_dialogue: str = "",
) -> list[CompanionIntentResult]:
    raw = (text or "").strip()
    if not raw:
        return [
            _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本"),
        ]
    mapped = _rewrite_synonyms(raw)
    query = mapped or raw
    items = classify_companion_intent_llm(
        model if model is not None else _default_chat_model(),
        query,
        recent_dialogue=recent_dialogue,
    )
    if items:
        return [_prefer_symptom_care(item, raw) for item in items]
    llm_model = model if model is not None else _default_chat_model()
    rows = [_prefer_symptom_care(item, raw) for item in classify_intent_fallback_multi(raw)]
    rows = _ensure_major_scores(rows, query)
    scores = dict(rows[0].major_scores) if rows and rows[0].major_scores else (_predict_major_scores_bert(query) or {})
    rows = _reconcile_intents_with_major_scores(rows, scores, raw, llm_model)
    rows = normalize_intent_results(rows)
    return rows


def classify_companion_intent(
    text: str,
    model: Any = None,
    *,
    recent_dialogue: str = "",
) -> CompanionIntentResult:
    items = classify_companion_intents(text, model, recent_dialogue=recent_dialogue)
    return items[0]


def resolve_companion_intent_items(
    text: str,
    model: Any = None,
    *,
    recent_dialogue: str = "",
    backend: str = "llm",
) -> list[CompanionIntentResult]:
    """意图解析子步；完整进线请用 algorithm.companion_turn.resolve_companion_turn。"""
    mode = (backend or "llm").strip().lower()
    if mode == "bert":
        return classify_companion_intents_bert(text)
    return classify_companion_intents(text, model, recent_dialogue=recent_dialogue)
