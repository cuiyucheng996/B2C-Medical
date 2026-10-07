---
handler: report_explain
major: 报告指标解读
intent: 解读用户提到的检查报告
---

## 目标
解读用户口头提到的化验/检查（非刚上传 PDF）。

## 执行步骤
1. 只调 explain_mentioned_report；exam 填检查名。

## 禁止
- 不要 explain_uploaded_report / compare_lab_history。
