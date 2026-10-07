"""医疗问诊 FAQ 权威数据源（仅数据 + 离线匹配算法，不含 Redis 逻辑）。

维护说明
--------
- **只改本文件的** ``MEDICAL_FAQ_ITEMS`` 列表。
- **写入 Redis**：调用 ``redis_data_sync.sync_medical_faq_to_redis``（或启动时 ``medical_faq_store.bootstrap_medical_faq``）。
- **运行时读取**：``medical_faq_store.match_medical_faq``（数据来自 Redis 缓存）。

勿在本文件内直接操作 Redis；键约定与 pipeline 均在 ``redis_data_sync.py``。
"""

from __future__ import annotations

import re

from src.schemas.medical_faq import FaqPhase, MedicalFaqItem

MEDICAL_FAQ_ITEMS: list[MedicalFaqItem] = [
    # --- chitchat ---
    {
        "id": "chitchat_hello",
        "question": "你好 / 在吗？",
        "answer": "您好，我是中医问诊助手。请告诉我最近哪里不舒服，或先选择对应部位。",
        "category": "chitchat",
        "keywords": "你好,您好,在吗,有人吗,hi,hello",
        "phase": "any",
        "priority": 90,
        "action": "reply_and_continue",
    },
    {
        "id": "chitchat_thanks",
        "question": "谢谢 / 感谢",
        "answer": "不客气。若还有症状需要补充，请继续告诉我。",
        "category": "chitchat",
        "keywords": "谢谢,感谢,多谢,辛苦了",
        "phase": "any",
        "priority": 85,
        "action": "reply_only",
    },
    {
        "id": "chitchat_goodbye",
        "question": "再见 / 不问了",
        "answer": "好的，如有需要随时再来。若不适加重，请及时线下就医或拨打 120。",
        "category": "chitchat",
        "keywords": "再见,拜拜,不问了,算了,不用了",
        "phase": "any",
        "priority": 80,
        "action": "reply_only",
    },
    {
        "id": "chitchat_who_are_you",
        "question": "你是谁 / 你能做什么？",
        "answer": "我是中医问诊助手，帮您梳理不适部位与症状类型，辅助分析。我不能替代线下面诊。请说说哪里不舒服？",
        "category": "chitchat",
        "keywords": "你是谁,你是什么,你能做什么,你会什么,机器人,人工智能,ai",
        "phase": "any",
        "priority": 88,
        "action": "reply_and_continue",
    },
    {
        "id": "chitchat_filler",
        "question": "嗯 / 啊 / 哦（语气词）",
        "answer": "没关系，您可以慢慢说。请描述一下最近哪里不太舒服？",
        "category": "chitchat",
        "keywords": "嗯,啊,哦,呃,额,唉",
        "phase": "any",
        "priority": 50,
        "action": "reply_and_continue",
    },
    # --- usage ---
    {
        "id": "usage_how_to_use",
        "question": "怎么用 / 如何使用？",
        "answer": "请先选择最相关的不适部位；再选择或描述疾病类型；确认后我会继续问具体症状。您也可以直接用文字描述。",
        "category": "usage",
        "keywords": "怎么用,如何使用,怎么操作,如何使用你,怎么开始",
        "phase": "any",
        "priority": 95,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_other_option",
        "question": "「其他」是什么意思？",
        "answer": "「找不到对应部位」表示选项里没有完全符合的，请用文字描述您的具体感受，我来帮您分析。",
        "category": "usage",
        "keywords": "其他是什么意思,找不到对应,找不到部位,其他选项",
        "phase": "body",
        "priority": 92,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_change_selection",
        "question": "选错了怎么办？",
        "answer": "可以直接重新点选部位或疾病，也可以用文字说明正确的情况，我会按最新描述继续。",
        "category": "usage",
        "keywords": "选错了,改一下,重新选,换一下,选错",
        "phase": "any",
        "priority": 88,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_text_input",
        "question": "可以打字吗 / 必须点按钮吗？",
        "answer": "可以。部位和疾病既支持点选，也支持直接输入文字描述您的症状。",
        "category": "usage",
        "keywords": "可以打字吗,必须点吗,必须选择吗,能不能输入,打字",
        "phase": "any",
        "priority": 86,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_voice",
        "question": "能语音吗 / 怎么语音输入？",
        "answer": "语音功能即将接入。目前请使用文字输入或点选按钮；接入后可在左侧麦克风区域说话。",
        "category": "usage",
        "keywords": "语音,说话,麦克风,语音输入,语音怎么用",
        "phase": "any",
        "priority": 84,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_must_select_disease",
        "question": "一定要选疾病吗？",
        "answer": "建议先锁定一个最贴近的类型，便于后续追问具体症状。若都不符合，选「其他」并文字描述即可。",
        "category": "usage",
        "keywords": "一定要选疾病吗,必须选疾病,可以不选吗,跳过疾病",
        "phase": "disease",
        "priority": 90,
        "action": "reply_and_continue",
    },
    {
        "id": "usage_what_to_do_now",
        "question": "现在我该做什么？",
        "answer": "请根据页面提示：选部位、选或描述疾病，或补充具体症状。不确定就从「哪里不舒服」说起即可。",
        "category": "usage",
        "keywords": "现在该做什么,接下来怎么办,下一步,我该干嘛",
        "phase": "any",
        "priority": 87,
        "action": "reply_and_continue",
    },
    # --- process ---
    {
        "id": "process_why_body_part",
        "question": "为什么要选部位？",
        "answer": "先确定不适大致涉及的身体区域，才能缩小可能的疾病类型，后面问得更有针对性。",
        "category": "process",
        "keywords": "为什么要选部位,为何选部位,选部位干什么",
        "phase": "body",
        "priority": 82,
        "action": "reply_and_continue",
    },
    {
        "id": "process_why_disease",
        "question": "为什么要选疾病？",
        "answer": "锁定疾病类型后，我会按该类型常见症状继续追问，比泛泛聊天更准确。",
        "category": "process",
        "keywords": "为什么要选疾病,为何选疾病,选疾病干什么",
        "phase": "disease",
        "priority": 82,
        "action": "reply_and_continue",
    },
    {
        "id": "process_consent",
        "question": "确认继续分析是什么意思？",
        "answer": "表示您同意在已有信息基础上进入下一轮，我会进一步询问具体症状和细节。",
        "category": "process",
        "keywords": "确认继续,继续分析,同意继续,可以继续吗",
        "phase": "consent",
        "priority": 90,
        "action": "reply_and_continue",
    },
    {
        "id": "process_rounds",
        "question": "一共要问几轮？",
        "answer": "整体分四大轮：先定部位与疾病类型，再锁症状、补信息、快捷补充。当前处于第一轮。",
        "category": "process",
        "keywords": "几轮,多少轮,问多久,要聊多久,分几步",
        "phase": "any",
        "priority": 78,
        "action": "reply_and_continue",
    },
    # --- disclaimer ---
    {
        "id": "disclaimer_not_diagnosis",
        "question": "这是诊断吗 / 准不准？",
        "answer": "这是辅助问诊梳理，不能作为最终诊断或处方依据。请以正规医疗机构面诊为准。",
        "category": "disclaimer",
        "keywords": "诊断吗,是看病吗,准不准,准确吗,能确诊吗,是不是病",
        "phase": "any",
        "priority": 98,
        "action": "disclaimer",
    },
    {
        "id": "disclaimer_no_prescription",
        "question": "能不能开药 / 吃什么药？",
        "answer": "我不能开具处方或推荐具体用药。请携带症状描述到正规医疗机构，由医生面诊后决定。",
        "category": "disclaimer",
        "keywords": "开药,吃什么药,用什么药,药方,处方,推荐药",
        "phase": "any",
        "priority": 98,
        "action": "disclaimer",
    },
    {
        "id": "disclaimer_severity",
        "question": "严不严重 / 要不要去医院？",
        "answer": "我无法远程判断严重程度。若出现胸痛、呼吸困难、意识不清、大量出血等，请立即拨打 120 或急诊就医。",
        "category": "disclaimer",
        "keywords": "严不严重,严重吗,要不要去医院,要去医院吗,要紧吗,危险吗",
        "phase": "any",
        "priority": 97,
        "action": "disclaimer",
    },
    {
        "id": "disclaimer_privacy",
        "question": "隐私 / 数据会保存吗？",
        "answer": "会话用于本次问诊流程。具体存储策略以产品隐私说明为准；请勿输入身份证号、银行卡等敏感个人信息。",
        "category": "disclaimer",
        "keywords": "隐私,数据保存,会记录吗,保密,个人信息",
        "phase": "any",
        "priority": 75,
        "action": "disclaimer",
    },
    {
        "id": "disclaimer_fee",
        "question": "收费吗 / 多少钱？",
        "answer": "当前为问诊辅助体验，是否收费以产品正式说明为准。",
        "category": "disclaimer",
        "keywords": "收费吗,多少钱,免费吗,价格,费用",
        "phase": "any",
        "priority": 70,
        "action": "disclaimer",
    },
    # --- knowledge ---
    {
        "id": "knowledge_body_parts",
        "question": "有哪些部位可以选？",
        "answer": "可选：头面部、颈部、胸胁部、胃腹部、二阴部、四肢躯干、其他，以及头面/四肢相关的黄疸与水肿。",
        "category": "knowledge",
        "keywords": "有哪些部位,部位有哪些,六大部位,八大部位,选什么部位",
        "phase": "body",
        "priority": 80,
        "action": "reply_and_continue",
    },
    {
        "id": "knowledge_head_face_diseases",
        "question": "头面部包括哪些疾病？",
        "answer": "头面部常见类型包括：头痛、眩晕、中风、痫病、痴呆、鼻衄、齿衄、咳血、吐血等。请选择最贴近的一项或文字描述。",
        "category": "knowledge",
        "keywords": "头面部有哪些,头面部包括,头面部疾病",
        "phase": "disease",
        "priority": 76,
        "action": "reply_and_continue",
    },
    {
        "id": "knowledge_symptom_vs_disease",
        "question": "症状和疾病有什么区别？",
        "answer": "症状是您感受到的表现（如头疼、乏力）；疾病类型是中医归类方向，便于后续系统追问。请先选最接近的类型。",
        "category": "knowledge",
        "keywords": "症状和疾病,有什么区别,什么是症状,什么是疾病",
        "phase": "any",
        "priority": 72,
        "action": "reply_and_continue",
    },
    # --- boundary ---
    {
        "id": "boundary_lab_report",
        "question": "帮看化验单 / 检查报告",
        "answer": "目前不支持解读化验单或影像报告。请携带资料到医疗机构，由医生结合面诊判断。",
        "category": "boundary",
        "keywords": "化验单,检查报告,体检报告,报告单,单子帮我看",
        "phase": "any",
        "priority": 96,
        "action": "disclaimer",
    },
    {
        "id": "boundary_insist_diagnosis",
        "question": "你就说是啥病",
        "answer": "在信息不足时我不能给出确定病名。请补充：哪里不舒服、持续多久、什么情况下加重或缓解。",
        "category": "boundary",
        "keywords": "就说是啥病,直接告诉我,别问了,结论呢",
        "phase": "any",
        "priority": 94,
        "action": "disclaimer",
    },
    {
        "id": "boundary_off_topic",
        "question": "与问诊无关的话题（兜底）",
        "answer": "我主要负责身体不适与问诊相关的问题。若需健康咨询，请描述您的症状或选择对应部位。",
        "category": "boundary",
        "keywords": "天气,股票,新闻,游戏,政治",
        "phase": "any",
        "priority": 40,
        "action": "reply_and_continue",
    },
]


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def match_faq_items(
    items: list[MedicalFaqItem],
    query: str,
    *,
    phase: FaqPhase | str = "any",
) -> MedicalFaqItem | None:
    """对给定 FAQ 列表做关键词匹配（返回最佳一条）。"""
    ranked = suggest_faq_items(items, query, phase=phase, limit=1)
    return ranked[0] if ranked else None


def _faq_item_score(item: MedicalFaqItem, compact: str) -> int:
    best = -1
    question_norm = _normalize_query(item["question"])

    if compact and question_norm and (compact in question_norm or question_norm in compact):
        best = max(best, item["priority"] + min(len(compact), len(question_norm)))

    for kw in item["keywords"].split(","):
        keyword = _normalize_query(kw)
        if not keyword:
            continue
        if keyword in compact or compact in keyword:
            best = max(best, item["priority"] + len(keyword))

    return best


def suggest_faq_items(
    items: list[MedicalFaqItem],
    query: str,
    *,
    phase: FaqPhase | str = "any",
    limit: int = 5,
) -> list[MedicalFaqItem]:
    """关键词 / 标准问法联想，按相关度返回多条（去重）。"""
    compact = _normalize_query(query)
    if not compact or limit < 1:
        return []

    current_phase = phase if phase in ("any", "body", "disease", "consent", "idle") else "any"
    ranked: list[tuple[int, MedicalFaqItem]] = []
    seen: set[str] = set()

    for item in items:
        item_phase = item["phase"]
        if item_phase not in ("any", current_phase) and current_phase != "any":
            continue
        score = _faq_item_score(item, compact)
        if score < 0:
            continue
        faq_id = item["id"]
        if faq_id in seen:
            continue
        seen.add(faq_id)
        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def match_medical_faq(
    query: str,
    *,
    phase: FaqPhase | str = "any",
) -> MedicalFaqItem | None:
    """离线/脚本用：直接匹配 ``MEDICAL_FAQ_ITEMS``（不经过 Redis）。"""
    return match_faq_items(MEDICAL_FAQ_ITEMS, query, phase=phase)


__all__ = [
    "MEDICAL_FAQ_ITEMS",
    "match_faq_items",
    "match_medical_faq",
    "suggest_faq_items",
]
