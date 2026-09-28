"""指代补全：纠错之后、意图之前，把「刚才说的/该/那个」补成明确名词。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

__all__ = ["CorefRewrite", "rewrite_coref_query"]

logger = logging.getLogger(__name__)

_CUE_RE = re.compile(
    r"("
    r"刚才说的那个病|刚刚说的那个病|上面说的那个病|"
    r"刚才说的病|刚刚说的病|上面说的病|前面说的病|上次说的病|"
    r"刚才说的那个|刚刚说的那个|上面说的那个|"
    r"刚才说的|刚刚说的|上面说的|前面说的|上次说的|"
    r"这个疾病|那个疾病|这个病|那个病|这病|那病|该病|"
    r"这个症状|那个症状|该症状|"
    r"这个部位|那个部位|这个地方|那个地方|该部位|"
    r"这个检查|那个检查|这个指标|那个指标"
    r")"
)

_IT_RE = re.compile(r"(它)")

# 没有「这个/那个」时，短跟进问也要接上文（如「那危险吗」）
_FOLLOW_HEAD_RE = re.compile(r"^(那|那么|然后|还有)(?!边|里|个地方)")
_SHORT_ASK_RE = re.compile(
    r"^(危险吗|严重吗|正常吗|怎么办|怎么治|怎么看|能跑步吗|能运动吗|可以跑步吗|"
    r"要注意什么|注意什么|忌口吗|偏高怎么办|偏低怎么办)$"
)
_FOLLOW_TAIL_RE = re.compile(
    r"(会怎样|怎么样|怎么办|怎么治|怎么看|危险吗|严重吗|正常吗|"
    r"能跑步吗|能运动吗|可以.*吗|要注意什么|注意什么|忌口|偏高|偏低)"
)
_NEW_TOPIC_RE = re.compile(r"(最近|昨天|今天|这几天|我有|我这|失眠|下雨|吃饭)")
_LAB_RE = re.compile(
    r"(ALT|AST|GGT|HBsAg|TSH|转氨酶|谷丙|谷草|血糖|胆固醇|甘油三酯|"
    r"白细胞|红细胞|血红蛋白|肌酐|尿素氮|甲功|血常规|肝功能|尿常规)",
    re.I,
)
_SLOT_KEYS = ("指标", "检查", "疾病", "症状", "部位")


@dataclass(frozen=True)
class CorefRewrite:
    text: str
    changed: bool
    fillers: dict[str, str]
    reason: str
    source: str = ""


def rewrite_coref_query(
    text: str,
    *,
    last_preprocess: dict[str, Any] | None = None,
    last_user_text: str = "",
    slot_memory: dict[str, list[str]] | None = None,
    prior_user_texts: list[str] | None = None,
    allow_llm: bool = True,
) -> CorefRewrite:
    raw = (text or "").strip()
    if not raw:
        return CorefRewrite("", False, {}, "空文本")
    fillers = _fillers_from_session(slot_memory, last_preprocess)
    prior = [str(item).strip() for item in (prior_user_texts or []) if str(item).strip()]
    if not prior and (last_user_text or "").strip():
        prior = [(last_user_text or "").strip()]
    if prior:
        fillers["上文"] = "；".join(prior[-5:])[:200]
        _fill_topic_from_prior(fillers, prior[-1])
    if not _needs_coref(raw):
        return CorefRewrite(raw, False, {}, "无指代", source="none")
    has_slot = bool(_topic(fillers))
    if not has_slot and not fillers.get("上文"):
        return CorefRewrite(raw, False, {}, "有指代但本场尚无槽位", source="none")

    replaced = _fill_by_slots(raw, fillers)
    if replaced != raw and has_slot:
        kept = {key: value for key, value in fillers.items() if value}
        return CorefRewrite(replaced, True, kept, "已按本场会话槽位补全", source="slot")

    if allow_llm and (not has_slot or replaced == raw):
        llm_text = _llm_complete(raw, prior)
        if llm_text and llm_text != raw:
            kept = {key: value for key, value in fillers.items() if value}
            return CorefRewrite(llm_text, True, kept, "槽位未抽出指代对象，已用模型补全", source="llm")

    if fillers.get("上文"):
        kept = {key: value for key, value in fillers.items() if value}
        return CorefRewrite(
            f"{raw}（承接上文：{fillers['上文']}）",
            True,
            kept,
            "模型未补全，挂接上文",
            source="context_tail",
        )
    return CorefRewrite(raw, False, fillers, "指代未替换", source="none")


def _needs_coref(text: str) -> bool:
    compact = (text or "").replace(" ", "")
    if _CUE_RE.search(text) or _CUE_RE.search(compact):
        return True
    if "它" in compact and len(compact) <= 24:
        return True
    if _is_followup(text):
        return True
    return False


def _is_followup(text: str) -> bool:
    compact = (text or "").replace(" ", "")
    if not compact or len(compact) > 22:
        return False
    if _NEW_TOPIC_RE.search(compact):
        return False
    if _SHORT_ASK_RE.match(compact):
        return True
    if _FOLLOW_HEAD_RE.match(compact) and _FOLLOW_TAIL_RE.search(compact):
        return True
    return False


def _fillers_from_session(
    slot_memory: dict[str, list[str]] | None,
    last_preprocess: dict[str, Any] | None,
) -> dict[str, str]:
    out: dict[str, str] = {key: "" for key in _SLOT_KEYS}
    memory = slot_memory if isinstance(slot_memory, dict) else {}
    for key in _SLOT_KEYS:
        items = [str(item).strip() for item in (memory.get(key) or []) if str(item).strip()]
        if items:
            out[key] = items[0]
    last = _fillers_from_preprocess(last_preprocess)
    for key in _SLOT_KEYS:
        out[key] = last.get(key) or out[key]
    return out


def _fillers_from_preprocess(payload: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {key: "" for key in _SLOT_KEYS}
    if not isinstance(payload, dict):
        return out
    mentions = payload.get("mentions") if isinstance(payload.get("mentions"), dict) else {}
    for key in _SLOT_KEYS:
        items = mentions.get(key) or []
        if items:
            out[key] = str(items[0]).strip()
    for hit in payload.get("alignments") or []:
        if not isinstance(hit, dict):
            continue
        name = str(hit.get("standard_name") or hit.get("mention") or "").strip()
        if not name:
            continue
        label = str(hit.get("node_label") or "")
        if "Disease" in label:
            out["疾病"] = name
        elif "Symptom" in label:
            out["症状"] = out["症状"] or name
        elif "BodyPart" in label:
            out["部位"] = out["部位"] or name
    return out


def _fill_topic_from_prior(fillers: dict[str, str], last_user: str) -> None:
    text = (last_user or "").strip()
    if not text:
        return
    lab = _LAB_RE.search(text)
    if lab and not fillers.get("指标"):
        fillers["指标"] = lab.group(1)


def _fill_by_slots(raw: str, fillers: dict[str, str]) -> str:
    replaced = _replace_cues(raw, fillers)
    if replaced == raw and "它" in raw:
        topic = _topic(fillers)
        if topic:
            replaced = _IT_RE.sub(topic, raw, count=1)
    if replaced == raw and _is_followup(raw):
        topic = _topic(fillers)
        if topic and topic not in raw:
            replaced = f"{topic}：{raw}"
    return replaced


def _llm_complete(query: str, prior: list[str]) -> str:
    history = "\n".join(f"用户：{item}" for item in prior[-6:] if item)
    if not history:
        return ""

    def _call() -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.llm.chat import get_chat_model

        llm = get_chat_model()
        if llm is None:
            return ""
        invoke_model = llm
        if hasattr(llm, "model_copy"):
            try:
                invoke_model = llm.model_copy(update={"timeout": 6.0})
            except Exception:
                invoke_model = llm
        from algorithm.llm_trace import STAGE_COREF, record_llm_call, serialize_langchain_messages

        msgs = [
            SystemMessage(
                content=(
                    "你做指代补全。根据对话历史，把当前用户这句话改写成一句独立、可检索的中文问句，"
                    "必须写明历史里的指代对象（可以是概念、指标、书名、很长一截说法）。"
                    "只输出改写后的问句，不要解释。若无法补全，原样输出当前这句话。"
                )
            ),
            HumanMessage(content=f"历史：\n{history}\n\n当前：{query}"),
        ]
        resp = invoke_model.invoke(msgs)
        text = str(getattr(resp, "content", "") or "").strip().splitlines()[0].strip(" \"'`")
        record_llm_call(
            STAGE_COREF,
            serialize_langchain_messages(msgs),
            response=text,
            label="coref _llm_complete",
        )
        return text

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    from algorithm.llm_trace import run_in_llm_trace_context

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return str(
            pool.submit(lambda: run_in_llm_trace_context(_call)).result(timeout=6) or ""
        ).strip()
    except FuturesTimeout:
        logger.warning("coref llm timeout")
        return ""
    except Exception:
        logger.warning("coref llm failed", exc_info=True)
        return ""
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _topic(fillers: dict[str, str]) -> str:
    return _prefer(fillers, *_SLOT_KEYS)


def _prefer(fillers: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = (fillers.get(key) or "").strip()
        if value:
            return value
    return ""


def _replace_cues(text: str, fillers: dict[str, str]) -> str:
    disease = _prefer(fillers, "疾病")
    symptom = _prefer(fillers, "症状")
    part = _prefer(fillers, "部位")
    metric = _prefer(fillers, "指标")
    exam = _prefer(fillers, "检查")
    fallback = metric or exam or disease or symptom or part

    mapping: list[tuple[str, str]] = []
    if disease:
        mapping.extend(
            [
                ("刚才说的那个病", disease),
                ("刚刚说的那个病", disease),
                ("上面说的那个病", disease),
                ("刚才说的病", disease),
                ("刚刚说的病", disease),
                ("上面说的病", disease),
                ("前面说的病", disease),
                ("上次说的病", disease),
                ("这个疾病", disease),
                ("那个疾病", disease),
                ("这个病", disease),
                ("那个病", disease),
                ("该病", disease),
                ("这病", disease),
                ("那病", disease),
            ]
        )
    if symptom:
        mapping.extend(
            [
                ("这个症状", symptom),
                ("那个症状", symptom),
                ("该症状", symptom),
            ]
        )
    if part:
        mapping.extend(
            [
                ("这个部位", part),
                ("那个部位", part),
                ("这个地方", part),
                ("那个地方", part),
                ("该部位", part),
            ]
        )
    if metric:
        mapping.extend(
            [
                ("这个指标", metric),
                ("那个指标", metric),
                ("该指标", metric),
            ]
        )
    if exam:
        mapping.extend(
            [
                ("这个检查", exam),
                ("那个检查", exam),
                ("该检查", exam),
            ]
        )
    if fallback:
        mapping.extend(
            [
                ("刚才说的那个", fallback),
                ("刚刚说的那个", fallback),
                ("上面说的那个", fallback),
                ("刚才说的", fallback),
                ("刚刚说的", fallback),
                ("上面说的", fallback),
                ("前面说的", fallback),
                ("上次说的", fallback),
            ]
        )
    mapping.sort(key=lambda item: len(item[0]), reverse=True)
    out = text
    for cue, value in mapping:
        if cue in out:
            return out.replace(cue, value, 1)
    return out
