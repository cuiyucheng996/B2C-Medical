"""生成 C 端七类大意图 + other 语料（text,intent），格式对齐老师 data.csv。"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
B2B_SRC = Path(r"D:\program\B2B") / "src"
if str(B2B_SRC) not in sys.path:
    sys.path.insert(0, str(B2B_SRC))
DATA_DIR = B2C_ROOT / "data" / "intent_major"
OUTPUT_CSV = DATA_DIR / "data.csv"
OTHER_CSV = DATA_DIR / "other_data.csv"

MAJOR_SAMPLES: dict[str, tuple[str, ...]] = {
    "公共知识问答": (
        "胃炎有哪些症状",
        "头部常见有哪些病",
        "血常规这个检查是查什么的",
        "转氨酶正常值是多少",
        "脂肪肝通俗讲是什么病",
        "头痛要注意什么忌口吗",
        "B超是看什么的",
        "脾虚常见有哪些表现",
        "高血压典型症状是什么",
        "咳嗽能不能运动",
        "胃镜检查什么",
        "这个指标参考范围呢",
        "腰椎间盘突出会有什么表现",
        "失眠饮食注意什么",
        "心悸通常在哪里不舒服",
        "肝郁是什么证型表现",
        "感冒会有什么表现",
        "腰部常见疾病有哪些",
        "偏高是什么意思这个指标",
        "解释一下这个病叫失眠",
    ),
    "具体任务": (
        "提醒我下周复查",
        "帮我设个随访提醒",
        "今天药打卡了",
        "完成打卡",
        "到时候复查提醒我一声",
        "记下我喝过药了",
        "提醒我明天去医院",
        "帮我设个闹钟随访",
        "今日打卡成功",
        "别忘了提醒我复查",
    ),
    "个人记忆问答": (
        "我说过哪里不舒服",
        "还记得我上次说的病吗",
        "我最近心情你还记得吗",
        "还记得我作业没有吗",
        "上次去学校你记得吗",
        "我提过什么部位",
        "你还记得我说过头疼吗",
        "我以前说过什么病",
        "记忆里我最近开心过吗",
        "查一下我说过的不适",
    ),
    "报告指标解读": (
        "帮我看看这份化验单",
        "报告怎么看",
        "转氨酶和上次比怎么样",
        "这个指标比上次高吗",
        "帮我看看检查报告",
        "指标变化大吗",
        "化验单上这项什么意思",
        "对比一下上次的血常规",
        "报告里的箭头是什么意思",
        "帮我解读一下体检单",
        "我要上传检查报告请解读",
        "上传化验单帮我看看",
    ),
    "风险拦截": (
        "快救命",
        "要叫救护车吗",
        "我晕倒了剧烈胸痛",
        "这个药一天几次",
        "能停药吗剂量怎么调",
        "吃几粒合适",
        "急救现在怎么办",
        "加大剂量可以吗",
        "胸痛喘不上气",
        "药物剂量帮我算一下",
    ),
    "近况陈述": (
        "我头疼好几天了",
        "胸口闷不舒服好几天",
        "最近胃痛",
        "今天在学校玩得很开心",
        "外面下雨了心情不好",
        "作业也没有今天很轻松",
        "这两天睡不着",
        "腰有点酸",
        "今天好累",
        "心情还可以就是有点烦",
        "脑袋疼好几天",
        "拉肚子了",
        "嗓子疼没劲",
        "胸口闷喘不上气有点吓人但不急",
    ),
    "闲聊": (
        "你好",
        "您好在吗",
        "hello",
        "谢谢",
        "再见",
        "辛苦了",
        "早上好",
        "哈哈好的",
        "嗯嗯",
        "打扰了",
    ),
}

OTHER_SAMPLES = (
    "今天股市怎么样",
    "帮我写首诗",
    "附近有什么好吃的",
    "怎么刷短视频",
    "明天下雨吗去哪玩",
    "推荐一部电影",
    "Python 怎么安装",
    "北京到上海机票",
    "今晚看什么综艺",
    "帮我算一道数学题",
)


_PREFIX = ("", "请问", "我想问", "帮我看看", "医生，")
_SUFFIX = ("", "？", "啊", "呢", "可以吗")


def _map_text(text: str) -> str:
    from schema.entity_synonyms import rewrite_synonyms_in_text

    return rewrite_synonyms_in_text(text) or text


def generate_rows(*, per_class: int = 120, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, samples in MAJOR_SAMPLES.items():
        for _ in range(per_class):
            text = f"{rng.choice(_PREFIX)}{rng.choice(samples)}{rng.choice(_SUFFIX)}".strip()
            rows.append((_map_text(text), intent))
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        writer.writerows(rows)


def main() -> None:
    rows = generate_rows()
    write_csv(OUTPUT_CSV, rows)
    write_csv(OTHER_CSV, [(_map_text(text), "other") for text in OTHER_SAMPLES] * 8)
    print(f"写入 {OUTPUT_CSV} 共 {len(rows)} 条，类别 {len(MAJOR_SAMPLES)}")
    print(f"写入 {OTHER_CSV}")


if __name__ == "__main__":
    main()
