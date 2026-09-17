"""具体任务细意图规则单测：避免宽泛词（提醒我）吃掉具体任务（吃药/复诊/取报告）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.intent_rule import match_intent, match_major_intent


class TaskIntentRuleTest(unittest.TestCase):
    def test_major_is_task(self) -> None:
        for text in ("提醒我吃药", "帮我挂号", "下周去拿报告", "提醒我复查"):
            self.assertEqual(match_major_intent(text), "具体任务", text)

    def test_medication_beats_generic_remind(self) -> None:
        for text in ("提醒我吃药", "该吃药了", "别忘了吃药", "设个服药提醒"):
            self.assertEqual(match_intent(text, major="具体任务"), "提醒用户按时服药", text)

    def test_appointment(self) -> None:
        for text in ("帮我挂号", "约个号", "下个月复诊", "预约复诊"):
            self.assertEqual(match_intent(text, major="具体任务"), "预约复诊或挂号", text)

    def test_report(self) -> None:
        for text in ("下周去拿报告", "提醒我取报告", "结果出来了"):
            self.assertEqual(match_intent(text, major="具体任务"), "提醒回传或领取检查报告", text)

    def test_remind_and_checkin_unchanged(self) -> None:
        self.assertEqual(match_intent("提醒我复查", major="具体任务"), "提醒用户某日复查或随访")
        self.assertEqual(match_intent("今天药打卡了", major="具体任务"), "记录用户完成一次打卡")

    def test_dosage_goes_to_risk_not_task(self) -> None:
        """剂量咨询属风险拦截，不能被服药提醒抢走。"""
        self.assertEqual(match_major_intent("这个药一天几次"), "风险拦截")


if __name__ == "__main__":
    unittest.main()
