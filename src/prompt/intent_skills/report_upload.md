---
handler: report_upload
major: 报告指标解读
intent: 上传检查报告并解读
---

## 目标
解读刚上传的 PDF 报告。

## 执行步骤
1. 只调 explain_uploaded_report。
2. 需要表格时再调 extract_report_tables（若本轮已绑定）。

## 禁止
- 不要 explain_mentioned_report / compare_lab_history。
