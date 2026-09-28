"""从用户话里抽出仍有效的生活事实（对齐 Graphiti fact，不写问诊节点）。"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from algorithm.life_lexicons import extract_life_mentions
from algorithm.talk_entities import extract_talk_entities
from prompt import load_prompt

logger = logging.getLogger(__name__)

__all__ = [
    "extract_companion_facts",
    "facts_from_slot_memory",
    "links_from_alignments",
    "links_from_slot_memory",
]

_FACT_KINDS = {
    "place",
    "time",
    "mood",
    "weather",
    "event",
    "occupation",
    "habit",
    "person",
    "thing",
    "negation",
    "other",
}

_KEEP_NEGATION = (
    "没有力气",
    "没力气",
    "没有精神",
    "没精神",
    "没有胃口",
    "没胃口",
    "睡不着",
    "不是很",
    "不是太",
    "不是特别",
)

_NEG_PREFIX = re.compile(r"(?:并不是|并没有|不是|没有|并非|并未)([^，。！？,.!?]{1,12})")

_LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ ~～]\S+)?$")
_ABS_YMD = re.compile(r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]")
_ABS_DASH = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")
_ABS_MD = re.compile(r"(?P<m>\d{1,2})月(?P<d>\d{1,2})[日号]")
_REL_DAYS = (
    ("大大前天", -4),
    ("大大后天", 4),
    ("大前天", -3),
    ("大后天", 3),
    ("前天", -2),
    ("后天", 2),
    ("昨天", -1),
    ("昨日", -1),
    ("明天", 1),
    ("明日", 1),
    ("今天", 0),
    ("今日", 0),
)
_YEAR_SHIFT = (
    ("前年", -2),
    ("去年", -1),
    ("今年", 0),
    ("明年", 1),
    ("后年", 2),
)
_MONTH_SHIFT = (
    ("上个月", -1),
    ("这个月", 0),
    ("本月", 0),
    ("下个月", 1),
)
_SAME_MOMENT = ("这个时候", "这个点儿", "这个点", "这时候", "这会儿")
_NUM_DAYS = re.compile(r"(?P<n>\d{1,3})天(?P<dir>前|后|以前|以后)")
_NUM_WEEKS = re.compile(r"(?P<n>\d{1,2})周(?P<dir>前|后|以前|以后)")
_WEEK_SHIFT = (("上上周", -2), ("下下周", 2), ("上周", -1), ("下周", 1), ("这周", 0), ("本周", 0))
_WEEKDAYS = (
    ("星期一", 0),
    ("星期二", 1),
    ("星期三", 2),
    ("星期四", 3),
    ("星期五", 4),
    ("星期六", 5),
    ("星期日", 6),
    ("星期天", 6),
    ("周一", 0),
    ("周二", 1),
    ("周三", 2),
    ("周四", 3),
    ("周五", 4),
    ("周六", 5),
    ("周日", 6),
    ("周天", 6),
)
_PERIODS = ("凌晨", "早上", "上午", "中午", "下午", "傍晚", "晚上", "夜里", "夜间")


def extract_companion_facts(
    text: str,
    *,
    at: str | datetime | None = None,
    skip_entity: bool = False,
) -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return []
    ref = _ref_local(at)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in _rule_facts(raw, ref):
        _put(merged, item)
    if not skip_entity:
        for item in _entity_facts(raw):
            _put(merged, item)
    for item in _llm_facts(raw, ref):
        if skip_entity and _looks_medical_fact(item):
            continue
        _put(merged, _normalize_time_fact(item, ref))
    return list(merged.values())


def _looks_medical_fact(item: dict[str, Any]) -> bool:
    blob = f"{item.get('name') or ''}{item.get('fact') or ''}"
    return any(token in blob for token in ("部位", "疾病", "症状", "body_part", "disease", "symptom"))


def facts_from_slot_memory(
    slot_memory: dict[str, list[str]] | None = None,
    mentions: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """把本场已抽出的部位/病/症写成 CompanionFact，不再关场时重新碰词表。"""
    labels = {"部位": "部位", "疾病": "疾病", "症状": "症状"}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for key in ("部位", "疾病", "症状"):
        names = list((slot_memory or {}).get(key) or []) + list((mentions or {}).get(key) or [])
        for name in names:
            span = str(name).strip()
            if not span or (key, span) in seen:
                continue
            seen.add((key, span))
            rows.append(
                {
                    "name": span,
                    "fact": f"用户提到{labels[key]}：{span}",
                    "kind": "other",
                    "negated": False,
                }
            )
    return rows


def links_from_slot_memory(
    slot_memory: dict[str, list[str]] | None = None,
    mentions: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """同一批实体写成 Finding 的 EXTRACTED 链接。"""
    mapping = {
        "部位": ("body_part", "ConsultBodyPart"),
        "疾病": ("disease", "ConsultDisease"),
        "症状": ("symptom", "ConsultPrimarySymptom"),
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for key, (kind, label) in mapping.items():
        names = list((slot_memory or {}).get(key) or []) + list((mentions or {}).get(key) or [])
        for name in names:
            span = str(name).strip()
            if not span or (kind, span) in seen:
                continue
            seen.add((kind, span))
            rows.append({"kind": kind, "name": span, "mention": span, "label": label})
    return rows


def links_from_alignments(alignments: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    mapping = {
        "ConsultBodyPart": ("body_part", "ConsultBodyPart"),
        "ConsultDisease": ("disease", "ConsultDisease"),
        "ConsultPrimarySymptom": ("symptom", "ConsultPrimarySymptom"),
        "BodyPart": ("body_part", "ConsultBodyPart"),
        "Disease": ("disease", "ConsultDisease"),
        "Symptom": ("symptom", "ConsultPrimarySymptom"),
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hit in alignments or []:
        if not isinstance(hit, dict):
            continue
        label = str(hit.get("node_label") or "")
        kind, node = None, None
        for key, mapped in mapping.items():
            if key in label:
                kind, node = mapped
                break
        name = str(hit.get("standard_name") or hit.get("mention") or "").strip()
        if not kind or not name or (kind, name) in seen:
            continue
        seen.add((kind, name))
        rows.append({"kind": kind, "name": name, "mention": name, "label": node})
    return rows


def _put(bucket: dict[tuple[str, str], dict[str, Any]], item: dict[str, Any]) -> None:
    name = str(item.get("name") or "").strip()
    fact = str(item.get("fact") or name).strip()
    kind = str(item.get("kind") or "other").strip()
    if kind not in _FACT_KINDS:
        kind = "other"
    if not name:
        return
    key = (kind, name)
    bucket[key] = {
        "name": name,
        "fact": fact,
        "kind": kind,
        "negated": bool(item.get("negated")),
    }


def _ref_local(at: str | datetime | None) -> datetime:
    if isinstance(at, datetime):
        dt = at if at.tzinfo else at.replace(tzinfo=_LOCAL_TZ)
        return dt.astimezone(_LOCAL_TZ)
    raw = (at or "").strip()
    if raw:
        try:
            naive = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
            return naive.replace(tzinfo=timezone.utc).astimezone(_LOCAL_TZ)
        except ValueError:
            pass
    return datetime.now(_LOCAL_TZ)


def _monday(day):
    return day - timedelta(days=day.weekday())


def _fmt_date(day, period: str = "") -> str:
    stamp = day.isoformat()
    return f"{stamp} {period}" if period else stamp


def _shift_years(day, years: int):
    try:
        return day.replace(year=day.year + years)
    except ValueError:
        return day.replace(year=day.year + years, day=28)


def _shift_months(day, months: int):
    idx = day.month - 1 + months
    year = day.year + idx // 12
    month = idx % 12 + 1
    try:
        return day.replace(year=year, month=month)
    except ValueError:
        return day.replace(year=year, month=month, day=28)


def _hit_longest(text: str, pairs: tuple[tuple[str, int], ...]) -> list[tuple[str, int]]:
    remaining = text
    hits: list[tuple[str, int]] = []
    for token, payload in pairs:
        if token in remaining:
            hits.append((token, payload))
            remaining = remaining.replace(token, "\u3000" * len(token))
    return hits


def _has_same_moment(text: str) -> bool:
    return any(token in text for token in _SAME_MOMENT)


def _period_near(text: str, token: str) -> str:
    idx = text.find(token)
    if idx < 0:
        return ""
    window = text[max(0, idx - 6) : idx + len(token) + 6]
    for period in _PERIODS:
        if period in window:
            return period
    return ""


def _time_facts(text: str, ref: datetime) -> list[dict[str, Any]]:
    compact = (text or "").replace(" ", "")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    today = ref.date()

    def add(name: str, mention: str) -> None:
        if not name or name in seen:
            return
        seen.add(name)
        rows.append(
            {
                "name": name,
                "fact": f"时间：{name}（原文：{mention}）",
                "kind": "time",
                "negated": False,
            }
        )

    for match in _ABS_YMD.finditer(compact):
        try:
            day = datetime(
                int(match.group("y")),
                int(match.group("m")),
                int(match.group("d")),
            ).date()
        except ValueError:
            continue
        add(_fmt_date(day, _period_near(compact, match.group(0))), match.group(0))
    for match in _ABS_DASH.finditer(compact):
        try:
            day = datetime(
                int(match.group("y")),
                int(match.group("m")),
                int(match.group("d")),
            ).date()
        except ValueError:
            continue
        add(_fmt_date(day, _period_near(compact, match.group(0))), match.group(0))

    year_hits = _hit_longest(compact, _YEAR_SHIFT)
    month_hits = _hit_longest(compact, _MONTH_SHIFT)
    same_moment = _has_same_moment(compact)
    clock = ref.strftime("%H:%M") if same_moment else ""
    year_delta = year_hits[0][1] if year_hits else 0
    md_found = False
    for match in _ABS_MD.finditer(compact):
        try:
            day = datetime(today.year, int(match.group("m")), int(match.group("d"))).date()
        except ValueError:
            continue
        if year_hits:
            day = _shift_years(day, year_delta)
        add(_fmt_date(day, _period_near(compact, match.group(0))), match.group(0))
        md_found = True

    for token, years in year_hits:
        if md_found:
            continue
        day = _shift_years(today, years)
        mention = token + ("这个时候" if same_moment else "")
        if same_moment:
            add(f"{day.isoformat()} {clock}", mention)
        else:
            add(_fmt_date(day, _period_near(compact, token)), token)

    for token, months in month_hits:
        day = _shift_months(today, months)
        mention = token + ("这个时候" if same_moment else "")
        if same_moment:
            add(f"{day.isoformat()} {clock}", mention)
        else:
            add(_fmt_date(day, _period_near(compact, token)), token)

    skip_today_token = bool(year_hits or month_hits)
    for token, delta in _hit_longest(compact, _REL_DAYS):
        if skip_today_token and token in {"今天", "今日"}:
            continue
        add(_fmt_date(today + timedelta(days=delta), _period_near(compact, token)), token)

    for match in _NUM_DAYS.finditer(compact):
        n = int(match.group("n"))
        sign = -1 if "前" in match.group("dir") else 1
        add(_fmt_date(today + timedelta(days=sign * n)), match.group(0))
    for match in _NUM_WEEKS.finditer(compact):
        n = int(match.group("n"))
        sign = -1 if "前" in match.group("dir") else 1
        add(_fmt_date(today + timedelta(weeks=sign * n)), match.group(0))

    if "刚才" in compact or "刚刚" in compact:
        mention = "刚才" if "刚才" in compact else "刚刚"
        add(ref.strftime("%Y-%m-%d %H:%M"), mention)
    elif same_moment and not year_hits and not month_hits:
        add(ref.strftime("%Y-%m-%d %H:%M"), "这个时候")

    for token, shift in _hit_longest(compact, _WEEK_SHIFT):
        start = _monday(today) + timedelta(weeks=shift)
        end = start + timedelta(days=6)
        weekday_hit = ""
        weekday_idx = -1
        for wtoken, widx in _WEEKDAYS:
            if wtoken in compact:
                weekday_hit = wtoken
                weekday_idx = widx
                break
        if weekday_idx >= 0:
            add(_fmt_date(start + timedelta(days=weekday_idx)), token + weekday_hit)
        else:
            add(f"{start.isoformat()}~{end.isoformat()}", token)

    if "最近" in compact and not year_hits:
        start = today - timedelta(days=6)
        add(f"{start.isoformat()}~{today.isoformat()}", "最近")

    if not any(token in compact for token, _ in _WEEK_SHIFT):
        for wtoken, widx in _WEEKDAYS:
            if wtoken not in compact:
                continue
            start = _monday(today)
            add(_fmt_date(start + timedelta(days=widx), _period_near(compact, wtoken)), wtoken)

    if not rows:
        for period in _PERIODS:
            if period in compact:
                add(_fmt_date(today, period), period)
                break
    return rows


def _normalize_time_fact(item: dict[str, Any], ref: datetime) -> dict[str, Any]:
    kind = str(item.get("kind") or "").strip()
    if kind != "time":
        return item
    name = str(item.get("name") or "").strip()
    if _DATE_RE.match(name) or re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}", name):
        return item
    blob = name + str(item.get("fact") or "")
    resolved = _time_facts(blob, ref)
    if not resolved:
        resolved = _time_facts(name, ref)
    if resolved:
        picked = resolved[0]
        return {
            **item,
            "name": picked["name"],
            "fact": picked["fact"],
            "kind": "time",
        }
    return {
        **item,
        "name": ref.date().isoformat(),
        "fact": f"时间：{ref.date().isoformat()}（原文：{name}）",
    }


def _entity_facts(text: str) -> list[dict[str, Any]]:
    """关场写入 CompanionFact 的部位/病/症 span，不依赖生活词表。"""
    try:
        from b2c_b2b_bridge import b2b_uie_extract

        extracted = b2b_uie_extract().extract_entity(text, ["部位", "疾病", "症状"])
    except Exception:
        logger.warning("companion entity facts skipped", exc_info=True)
        return []
    labels = {"部位": "部位", "疾病": "疾病", "症状": "症状"}
    rows: list[dict[str, Any]] = []
    for key, names in (extracted or {}).items():
        for name in names or []:
            span = str(name).strip()
            if not span:
                continue
            rows.append(
                {
                    "name": span,
                    "fact": f"用户提到{labels.get(key, key)}：{span}",
                    "kind": "other",
                    "negated": False,
                }
            )
    return rows


def _rule_facts(text: str, ref: datetime) -> list[dict[str, Any]]:
    compact = text.replace(" ", "")
    life = extract_life_mentions(text)
    labels = {"place": "地点", "mood": "心情", "weather": "天气"}
    rows: list[dict[str, Any]] = []
    for kind, names in life.items():
        if kind == "time":
            continue
        for name in names:
            rows.append(
                {
                    "name": name,
                    "fact": f"用户提到{labels.get(kind, kind)}：{name}",
                    "kind": kind if kind in _FACT_KINDS else "other",
                    "negated": False,
                }
            )
    rows.extend(_time_facts(text, ref))
    rows.extend(extract_talk_entities(text))
    keep = any(item in compact for item in _KEEP_NEGATION)
    if not keep:
        for match in _NEG_PREFIX.finditer(compact):
            span = (match.group(1) or "").strip("的了啊呀呢吧 ")
            if len(span) < 2:
                continue
            rows.append(
                {
                    "name": span,
                    "fact": f"用户否定：{span}",
                    "kind": "negation",
                    "negated": True,
                }
            )
    return rows


def _llm_facts(text: str, ref: datetime) -> list[dict[str, Any]]:
    try:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.llm.chat import get_chat_model

        model = get_chat_model()
        if model is None:
            return []
        reply = model.invoke(
            load_prompt("fact_extract").format(ref_date=ref.date().isoformat()) + text[:800]
        )
        content = getattr(reply, "content", "") or ""
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        payload = _parse_json(str(content))
        facts = payload.get("facts") if isinstance(payload, dict) else None
        if not isinstance(facts, list):
            return []
        return [item for item in facts if isinstance(item, dict)]
    except Exception:
        logger.warning("companion fact llm skipped")
        return []


def _parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if "```" in raw:
        raw = raw.split("```", 2)[1]
        raw = re.sub(r"^json", "", raw, flags=re.I).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
