"""意图：第三方模型同时判大类+细意图；失败再降级 BERT 大类 + 规则细类。"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from algorithm.intent_catalog import INTENT_BY_NAME, INTENT_ROWS, intent_prompt_lines, major_of
from algorithm.intent_rule import match_intent
from configs.paths_b2c import B2C_ROOT, b2b_src_imports, ensure_uvgraph_root
from prompt import load_prompt

logger = logging.getLogger(__name__)

__all__ = [
    "CompanionIntentResult",
    "classify_companion_intent",
    "classify_companion_intent_llm",
    "classify_fine_for_major",
    "classify_intent_fallback",
    "intent_result_to_dict",
    "normalize_intent_results",
]
_MAX_INTENTS = 3

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


def _major_model_dir() -> Path:
    return B2C_ROOT / "checkpoint" / "intent_major" / "best_model"


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


def _pack(
    *,
    major: str,
    intent: str,
    confidence: float,
    source: str,
    reason: str,
) -> CompanionIntentResult:
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


def normalize_intent_results(items: list[CompanionIntentResult]) -> list[CompanionIntentResult]:
    """去重、去掉被实义意图挤掉的闲聊/other，最多 3 条。"""
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
    real = [item for item in packed if item.major not in {"闲聊", "other"}]
    chosen = real or packed
    if not chosen:
        chosen = [_pack(major="other", intent="未知意图", confidence=0.0, source="empty", reason="无意图")]
    return chosen[:_MAX_INTENTS]


def _intent_catalog_prompt() -> str:
    return load_prompt("intent_catalog").format(
        majors="、".join(_MAJORS),
        intent_lines=intent_prompt_lines(),
    )


def _from_llm_fields(major: str, intent: str, reason: str) -> CompanionIntentResult:
    major = (major or "").strip()
    intent = (intent or "").strip()
    if intent in INTENT_BY_NAME:
        major = major_of(intent)
    elif intent != "未细分":
        intent = "未细分"
    if major not in _MAJORS:
        major = "other"
    return _pack(
        major=major,
        intent=intent if major != "other" else "未知意图",
        confidence=0.8,
        source="llm",
        reason=(reason or "").strip() or "llm",
    )


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
        major: str = Field(description="大类名称")
        intent: str = Field(description="细意图名称或未细分")
        reason: str = Field(description="短理由")

    class IntentChoices(BaseModel):
        items: list[IntentItem] = Field(description="本轮意图，1到3条")

    human = f"【近期对话】\n{recent_dialogue or '（无）'}\n\n【本轮用户】\n{text}"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        choice = _invoke_structured_output(
            model,
            IntentChoices,
            [SystemMessage(content=_intent_catalog_prompt()), HumanMessage(content=human)],
        )
    except Exception:
        logger.warning("llm intent classify failed", exc_info=True)
        return None
    raw_items = list(getattr(choice, "items", None) or [])
    packed = [
        _prefer_symptom_care(
            _fill_fine(
                _from_llm_fields(str(item.major or ""), str(item.intent or ""), str(item.reason or "")),
                query,
            ),
            query,
        )
        for item in raw_items
    ]
    if not packed:
        return None
    return normalize_intent_results(packed)


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
        resp = invoke_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=raw[:400])]
        )
        content = str(getattr(resp, "content", "") or "")
        match = _JSON_RE.search(content)
        if not match:
            raise ValueError("fine intent json missing")
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("fine intent json invalid")
        return str(data.get("intent") or "").strip(), str(data.get("reason") or "llm").strip()

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_call).result(timeout=seconds)
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
                packed = _pack(major=locked, intent=intent_name, confidence=0.85, source="llm", reason=reason)
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


def _predict_major_bert(text: str) -> tuple[str, float] | None:
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
        pred_id = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[pred_id].item())
    label = mapping.get(str(pred_id), "other")
    if label == "chitchat":
        label = "other"
    if label not in _MAJORS:
        label = "other"
    return label, round(confidence, 4)


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
        with b2b_src_imports():
            from schema.entity_synonyms import rewrite_synonyms_in_text

            mapped = rewrite_synonyms_in_text(text)
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
    if not fine:
        return item
    return _pack(
        major=item.major,
        intent=fine,
        confidence=item.confidence,
        source=f"{item.source}+rule",
        reason=(item.reason or "") + "；细意图规则补全",
    )


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

    bert = _predict_major_bert(query)
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
            )

    if major == "other":
        return _pack(major="other", intent="未知意图", confidence=confidence, source=source, reason=reason)

    fine = match_intent(raw, major=major) or match_intent(query, major=major)
    if fine:
        packed = _pack(major=major, intent=fine, confidence=confidence, source=source, reason=reason + "；细规则")
    else:
        packed = _pack(
            major=major,
            intent="未细分",
            confidence=confidence,
            source=source,
            reason=reason + "；细意图未命中",
        )
    return _prefer_symptom_care(packed, raw)


def classify_companion_intent(
    text: str,
    model: Any = None,
    *,
    recent_dialogue: str = "",
) -> CompanionIntentResult:
    raw = (text or "").strip()
    if not raw:
        return _pack(major="other", intent="未知意图", confidence=1.0, source="empty", reason="空文本")
    mapped = _rewrite_synonyms(raw)
    query = mapped or raw
    items = classify_companion_intent_llm(
        model if model is not None else _default_chat_model(),
        query,
        recent_dialogue=recent_dialogue,
    )
    if items:
        return _prefer_symptom_care(items[0], raw)
    return _prefer_symptom_care(classify_intent_fallback(raw), raw)
