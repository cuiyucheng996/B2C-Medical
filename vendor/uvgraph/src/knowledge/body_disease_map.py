"""八大部位与对应疾病列表（与问诊 UI / offer_disease_selection 一致）。"""

from __future__ import annotations

from ..schemas import BodyPart

BODY_PARTS: tuple[BodyPart, ...] = (
    "头面部",
    "颈部",
    "胸胁部",
    "胃腹部",
    "二阴部",
    "四肢躯干",
    "其他",
    "头面部、四肢躯干",
)

DISEASES_BY_BODY: dict[BodyPart, list[str]] = {
    "头面部": [
        "头痛",
        "眩晕",
        "中风",
        "痫病",
        "痴呆",
        "鼻衄",
        "齿衄",
        "咳血",
        "吐血",
    ],
    "颈部": ["喘证"],
    "胸胁部": ["胁痛", "心悸", "胸痹", "心衰"],
    "胃腹部": ["胃痛", "胃痞", "呕吐", "呃逆", "腹痛"],
    "二阴部": [
        "淋证",
        "癃闭",
        "阳痿",
        "遗精",
        "便血",
        "尿血",
        "泄泻",
        "痢疾",
        "便秘",
    ],
    "四肢躯干": ["痹证", "腰痛", "紫斑"],
    "其他": [
        "不寐",
        "消渴",
        "肥胖",
        "痰饮",
        "汗证",
        "内伤发热",
        "厥证",
        "虚劳",
        "郁证",
    ],
    "头面部、四肢躯干": ["黄疸", "水肿"],
}

BODY_PART_HINTS: dict[BodyPart, str] = {
    "头面部": "头部有头晕、头痛、中风、痴呆、鼻衄、咳血等症状",
    "颈部": "颈部气道呼吸急促、气短、喘促等症状",
    "胸胁部": "胸口闷痛、心慌、胁肋胀痛等症状",
    "胃腹部": "胃脘或腹部疼痛、胀满、呕吐、呃逆等症状",
    "二阴部": "排尿、排便、精液、月经等二阴相关异常",
    "四肢躯干": "四肢或躯干痹痛、腰痛、皮下紫斑等",
    "其他": "睡眠、形体、汗出、发热、情志、体虚等全身性不适",
    "头面部、四肢躯干": "头面或四肢皮肤巩膜发黄、小便发黄或周身浮肿",
}

__all__ = [
    "BODY_PARTS",
    "BODY_PART_HINTS",
    "DISEASES_BY_BODY",
    "diseases_for_body_part",
]


def diseases_for_body_part(body_part: str | None) -> list[str]:
    if not body_part:
        return []
    return list(DISEASES_BY_BODY.get(body_part, []))  # type: ignore[arg-type]
