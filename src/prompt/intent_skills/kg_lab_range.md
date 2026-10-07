---
handler: kg_lab_range
major: 公共知识问答
intent: 查询某指标的正常范围或含义
---

## 目标
说明指标参考范围或偏高/偏低含义。

## 执行步骤
1. 按绑定顺序先 search_exam_handbook，无证据再 search_chroma_knowledge。
2. 不要同时乱调 query_consult_graph。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}
