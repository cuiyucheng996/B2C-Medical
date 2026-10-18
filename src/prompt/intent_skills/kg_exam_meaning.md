---
handler: kg_exam_meaning
major: 公共知识问答
intent: 查询某检查项目是检查什么的
---

## 目标
说明检查/化验项目含义。

## 执行步骤
1. 按绑定顺序先 search_exam_handbook，无证据再 search_chroma_knowledge。
2. 用户只问检查含义时，不要调 query_consult_graph。
3. 用户问这项检查对应哪些病或部位时，手册之后可以再调一次 query_consult_graph，query 只填检查标准名。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}
