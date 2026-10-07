---
handler: kg_exam_meaning
major: 公共知识问答
intent: 查询某检查项目是检查什么的
---

## 目标
说明检查/化验项目含义。

## 执行步骤
1. 按绑定顺序先 search_exam_handbook，无证据再 search_chroma_knowledge。
2. 不要同时乱调 query_consult_graph。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}
