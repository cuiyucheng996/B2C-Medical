---
handler: kg_syndrome_signs
major: 公共知识问答
intent: 查询某证型常见有哪些表现
---

## 目标
说明证型常见表现。

## 执行步骤
1. 只调 query_consult_graph；query 填证型或相关检索词。

## 子句拆分（问诊图谱）
{{include:_snippets/graph_query_decompose.md}}
