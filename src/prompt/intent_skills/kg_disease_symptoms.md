---
handler: kg_disease_symptoms
major: 公共知识问答
intent: 查询某疾病常见有哪些症状
---

## 目标
列举某疾病常见症状/表现。

## 执行步骤
1. 只调 query_consult_graph；query 填标准疾病名。
2. 只用 data.items / summary。

## 子句拆分（问诊图谱）
{{include:_snippets/graph_query_decompose.md}}
