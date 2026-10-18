---
handler: kg_lab_range
major: 公共知识问答
intent: 查询某指标的正常范围或含义
---

## 目标
说明指标参考范围或偏高/偏低含义。

## 执行步骤
1. 按绑定顺序先 search_exam_handbook，无证据再 search_chroma_knowledge。
2. 用户只问参考范围或偏高偏低时，不要调 query_consult_graph。
3. 用户问该指标对应哪些病或部位时，手册之后可以再调一次 query_consult_graph，query 只填指标标准名。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}
