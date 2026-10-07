---
handler: kg_part_diseases
major: 公共知识问答
intent: 查询某部位有哪些常见疾病
---

## 目标
列举某部位常见疾病。

## 执行步骤
1. 只调 query_consult_graph；query 填标准部位名。
2. 只用 data.items / summary 组句。

## 子句拆分（问诊图谱）
{{include:_snippets/graph_query_decompose.md}}

## 禁止
- 不要调 search_exam_handbook / search_chroma_knowledge。
