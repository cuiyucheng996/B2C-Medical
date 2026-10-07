---
handler: kg_disease_plain
major: 公共知识问答
intent: 查询某疾病的通俗解释
---

## 目标
通俗解释某疾病。

## 执行步骤
1. 只调 query_consult_graph；query 填疾病名，侧重通俗说明。

## 子句拆分（问诊图谱）
{{include:_snippets/graph_query_decompose.md}}
