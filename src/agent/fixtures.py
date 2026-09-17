"""纠错回归样例（来自教材 + 项目预设）。"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SpellCheckSample", "SPELL_CHECK_SAMPLES"]


@dataclass(frozen=True, slots=True)
class SpellCheckSample:
    id: str
    name: str
    text: str
    expect: str


SPELL_CHECK_SAMPLES: tuple[SpellCheckSample, ...] = (
    SpellCheckSample(
        id="case1",
        name="新闻体拼音错误",
        text="小泉承诺将革除始曰苯陷于十年经济衰退的弊病。",
        expect="识别「始曰苯→使日本」类拼音输入错误",
    ),
    SpellCheckSample(
        id="case2",
        name="企业名混淆",
        text="万豪万家浙江冬日两家浙江老牌房企。",
        expect="修正为「万好万家浙江东日」类企业名称",
    ),
    SpellCheckSample(
        id="case3",
        name="语法错误（的→得）",
        text="我觉的每个人都应该买一个。",
        expect="识别「的→得」的语法错误",
    ),
    SpellCheckSample(
        id="case4",
        name="长句多错误",
        text="再加上在工作的地方有机会见面别的人，也可以学习新的文，最后经验越来越多。",
        expect="修正语病与用词（如「别的人」「新的文」）",
    ),
    SpellCheckSample(
        id="case5",
        name="形近字错误",
        text="根据最新数据显视，中国经济增长稳中有进。",
        expect="识别「显视→显示」",
    ),
    SpellCheckSample(
        id="case6",
        name="缺字补全",
        text="我喜你",
        expect="补全为「我喜欢你」或等价表达",
    ),
    SpellCheckSample(
        id="case7",
        name="无错误文本",
        text="今天天气很好。",
        expect="errors 为空，confidence 接近 1.0",
    ),
)
