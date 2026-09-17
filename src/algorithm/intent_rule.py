"""先大类、再细意图。细名只在已锁定的大类里选。"""

from __future__ import annotations

from algorithm.intent_catalog import INTENT_BY_NAME

INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("拦截急症或求救描述", ("救命", "急救", "叫救护车", "晕倒", "剧烈胸痛")),
    ("拦截具体用药剂量咨询", ("吃几粒", "一天几次", "剂量", "能停药吗", "加大剂量")),
    ("查询某检查项目是检查什么的", ("这个检查", "化验是查", "B超是看", "CT是看", "检查什么")),
    ("查询某指标的正常范围或含义", ("正常值", "参考范围", "偏高是什么", "偏低是什么", "这个指标")),
    ("查询某疾病的通俗解释", ("是什么病", "通俗讲", "解释一下这个病")),
    ("查询某证型常见有哪些表现", ("什么证", "证型表现", "肝郁常见", "脾虚常见")),
    (
        "查询某症状的生活注意事项",
        (
            "注意什么",
            "能不能运动",
            "饮食注意",
            "忌口",
            "注意事项",
            "怎么止血",
            "止血",
            "怎么护理",
            "怎么处理",
        ),
    ),
    ("查询某部位有哪些常见疾病", ("这个部位有哪些病", "头会得什么病", "常见疾病", "常见有哪些病", "会得什么病")),
    ("查询某疾病常见有哪些症状", ("有哪些症状", "会有什么表现", "典型症状")),
    ("查询某症状常见出现在哪些部位", ("出现在哪个部位", "是哪个地方的疼")),
    # 细意图按「越具体越靠前」排：服药/复诊/报告先于宽泛的「提醒我」
    ("提醒用户按时服药", ("该吃药", "吃药提醒", "服药提醒", "提醒我吃药", "别忘了吃药", "按时吃药")),
    ("预约复诊或挂号", ("帮我挂号", "挂个号", "预约挂号", "约个号", "预约复诊", "下个月复诊", "要复诊")),
    ("提醒回传或领取检查报告", ("取报告", "拿报告", "回传报告", "上传报告", "报告出来", "结果出来了")),
    ("提醒用户某日复查或随访", ("提醒我", "到时候复查", "随访提醒")),
    ("记录用户完成一次打卡", ("打卡了", "今天喝了药", "完成打卡")),
    ("查询用户曾经提到的部位或疾病", ("我说过哪里", "我上次说的病", "记得我哪里不舒服")),
    ("查询用户最近的心情或事件", ("我最近心情", "还记得我作业", "上次去学校")),
    ("上传检查报告并解读", ("上传检查报告", "上传化验单", "上传体检单", "pdf报告")),
    ("解读用户提到的检查报告", ("帮我看看报告", "化验单", "报告怎么看")),
    ("对比某指标与历史记录", ("和上次比", "比上次高", "指标变化")),
    ("解读用户上传的日常图片", ("上传图片", "【上传图片】", "看看这张图")),
    ("解读用户上传的普通文档", ("上传文档", "【上传文档】", "看看这个pdf")),
    ("陈述当前身体不适", ("我头疼", "不舒服好几天", "胸口闷")),
    ("陈述生活事件心情或天气", ("今天很开心", "下雨了", "去学校玩")),
    ("感谢或结束对话", ("谢谢", "再见", "辛苦了")),
    ("寒暄打招呼", ("你好", "您好", "在吗", "hello")),
]

# 大类判定顺序：越靠前越优先
MAJOR_ORDER = (
    "风险拦截",
    "报告指标解读",
    "个人记忆问答",
    "具体任务",
    "公共知识问答",
    "近况陈述",
    "闲聊",
)

__all__ = ["INTENT_RULES", "MAJOR_ORDER", "match_intent", "match_major_intent"]


def _compact(question: str) -> str:
    return (question or "").replace(" ", "").lower()


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw.replace(" ", "").lower() in text for kw in keywords if kw)


def match_major_intent(question: str) -> str | None:
    """只判七个大类，不返回细名。"""
    text = _compact(question)
    if not text:
        return None
    buckets: dict[str, list[str]] = {name: [] for name in MAJOR_ORDER}
    for intent, keywords in INTENT_RULES:
        major = str((INTENT_BY_NAME.get(intent) or {}).get("major") or "")
        if major in buckets:
            buckets[major].extend(keywords)
    for major in MAJOR_ORDER:
        if _hit(text, tuple(buckets[major])):
            return major
    return None


def match_intent(question: str, *, major: str | None = None) -> str | None:
    """在已锁定的大类内选细意图；未给大类则不匹配。"""
    if not major or major == "other":
        return None
    text = _compact(question)
    if not text:
        return None
    for intent, keywords in INTENT_RULES:
        row = INTENT_BY_NAME.get(intent) or {}
        if str(row.get("major") or "") != major:
            continue
        if _hit(text, keywords):
            return intent
    return None
