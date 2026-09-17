"""C 端生活向词表：心情、天气、地点、时间。不是问诊轮次槽。"""

from __future__ import annotations

__all__ = [
    "MOOD_HINTS",
    "PLACE_HINTS",
    "TIME_HINTS",
    "WEATHER_HINTS",
    "extract_life_mentions",
]

MOOD_HINTS = (
    "焦虑",
    "烦躁",
    "郁闷",
    "心情差",
    "心情不好",
    "开心",
    "紧张",
    "害怕",
    "睡不着",
    "失眠",
    "压力大",
    "累",
    "疲惫",
)

WEATHER_HINTS = (
    "下雨",
    "阴天",
    "大风",
    "台风",
    "很热",
    "很冷",
    "潮湿",
    "闷热",
    "降温",
    "下雪",
)

PLACE_HINTS = (
    "工地",
    "公司",
    "单位",
    "家里",
    "在家",
    "路上",
    "医院",
    "诊所",
    "办公室",
    "车间",
)

TIME_HINTS = (
    "大大前天",
    "大大后天",
    "大前天",
    "大后天",
    "前年",
    "去年",
    "今年",
    "明年",
    "后年",
    "上个月",
    "这个月",
    "下个月",
    "这个时候",
    "这时候",
    "前天",
    "后天",
    "昨天",
    "明天",
    "今天",
    "今日",
    "昨日",
    "明日",
    "刚才",
    "刚刚",
    "早上",
    "上午",
    "中午",
    "下午",
    "傍晚",
    "晚上",
    "夜里",
    "凌晨",
    "上周",
    "这周",
    "本周",
    "下周",
    "最近",
)


def extract_life_mentions(text: str) -> dict[str, list[str]]:
    compact = (text or "").replace(" ", "")
    found = {
        "mood": [item for item in MOOD_HINTS if item in compact],
        "weather": [item for item in WEATHER_HINTS if item in compact],
        "place": [item for item in PLACE_HINTS if item in compact],
        "time": [item for item in TIME_HINTS if item in compact],
    }
    return {key: list(dict.fromkeys(val)) for key, val in found.items()}
