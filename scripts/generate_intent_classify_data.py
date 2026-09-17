"""生成 C 端细意图训练语料（text,intent），格式对齐老师 data.csv。"""

from __future__ import annotations

import csv
import random
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = B2C_ROOT / "data" / "intent_classify"
OUTPUT_CSV = DATA_DIR / "data.csv"
OTHER_CSV = DATA_DIR / "other_data.csv"

TEMPLATES: dict[str, tuple[str, ...]] = {
    "查询某部位有哪些常见疾病": (
        "头部常见有哪些病",
        "胃这个部位会得什么病",
        "腰部常见疾病有哪些",
        "胸口这块容易有什么病",
    ),
    "查询某疾病常见有哪些症状": (
        "胃炎有哪些症状",
        "感冒会有什么表现",
        "高血压典型症状是什么",
        "腰椎间盘突出有哪些症状",
    ),
    "查询某症状常见出现在哪些部位": (
        "刺痛常见出现在哪个部位",
        "胀痛一般是哪个地方的",
        "心悸通常在哪里不舒服",
    ),
    "查询某疾病的通俗解释": (
        "胃炎是什么病",
        "通俗讲讲脂肪肝",
        "解释一下这个病叫失眠",
    ),
    "查询某检查项目是检查什么的": (
        "血常规这个检查是查什么的",
        "B超是看什么的",
        "胃镜检查什么",
    ),
    "查询某指标的正常范围或含义": (
        "转氨酶正常值是多少",
        "血压偏高是什么意思",
        "这个指标参考范围呢",
    ),
    "查询某症状的生活注意事项": (
        "头痛要注意什么",
        "失眠饮食注意什么",
        "咳嗽能不能运动",
        "发生创伤出血时怎么止血",
    ),
    "查询某证型常见有哪些表现": (
        "肝郁常见有哪些表现",
        "脾虚是什么证型表现",
        "湿热证常见表现",
    ),
    "提醒用户某日复查或随访": (
        "提醒我下周复查",
        "到时候随访提醒我一声",
        "帮我设个复查提醒",
    ),
    "记录用户完成一次打卡": (
        "今天药打卡了",
        "完成打卡",
        "今天喝了药",
    ),
    "查询用户曾经提到的部位或疾病": (
        "我说过哪里不舒服",
        "还记得我上次说的病吗",
        "我提过什么部位",
    ),
    "查询用户最近的心情或事件": (
        "我最近心情怎么样你还记得吗",
        "还记得我作业没有吗",
        "上次去学校你记得吗",
    ),
    "解读用户提到的检查报告": (
        "帮我看看这份化验单",
        "报告怎么看",
        "帮我看看检查报告",
    ),
    "对比某指标与历史记录": (
        "转氨酶和上次比怎么样",
        "这个指标比上次高吗",
        "指标变化大吗",
    ),
    "拦截急症或求救描述": (
        "快救命",
        "要叫救护车吗",
        "我晕倒了剧烈胸痛",
    ),
    "拦截具体用药剂量咨询": (
        "这个药一天几次",
        "能停药吗剂量怎么调",
        "吃几粒合适",
    ),
    "陈述当前身体不适": (
        "我头疼好几天了",
        "胸口闷不舒服好几天",
        "最近胃痛",
    ),
    "陈述生活事件心情或天气": (
        "今天在学校玩得很开心",
        "外面下雨了心情不好",
        "作业也没有今天很轻松",
    ),
    "寒暄打招呼": (
        "你好",
        "您好在吗",
        "hello",
    ),
    "感谢或结束对话": (
        "谢谢",
        "再见",
        "辛苦了",
    ),
}

OTHER_SAMPLES = (
    "今天股市怎么样",
    "帮我写首诗",
    "附近有什么好吃的",
    "怎么刷短视频",
)


def generate_rows(*, per_class: int = 40, seed: int = 42) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, samples in TEMPLATES.items():
        for _ in range(per_class):
            rows.append((rng.choice(samples), intent))
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
    write_csv(OTHER_CSV, [(text, "other") for text in OTHER_SAMPLES] * 10)
    print(f"写入 {OUTPUT_CSV} 共 {len(rows)} 条")
    print(f"写入 {OTHER_CSV}")


if __name__ == "__main__":
    main()
