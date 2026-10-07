---
handler: status_complaint
major: 近况陈述
intent: 陈述当前身体不适
---

## 目标
承接用户陈述当前身体不适，可补长期记忆。

## 执行步骤
1. 结合预处理观察与对话上文组句。
2. 需要时可调 search_long_memory 补长期近况。

## 禁止
- 不要调公共知识、报告、任务类工具。
