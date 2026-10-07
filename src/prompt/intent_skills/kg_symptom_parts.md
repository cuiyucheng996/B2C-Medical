---
handler: kg_symptom_parts
major: 公共知识问答
intent: 查询某症状常见出现在哪些部位
---

## 目标
说明某症状常见于哪些部位。

## 执行步骤
1. 只调 query_consult_graph；query 填症状名。

## 子句拆分（问诊图谱）
{{include:_snippets/graph_query_decompose.md}}
