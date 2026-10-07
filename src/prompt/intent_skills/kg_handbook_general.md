---
handler: kg_handbook_general
major: 公共知识问答
intent: 查询一般健康科普或常见问题
---

## 目标
健康科普、常见政策或行为类问答（如献血、疫苗、养生常识），从手册/向量库取依据。

## 执行步骤
1. 按绑定顺序先 search_exam_handbook，证据不足再 search_chroma_knowledge。
2. 不要调 query_consult_graph（非部位/病/症图谱问法）。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}

## 禁止
- 不要编造医学结论；手册无依据时说明未查到并建议就医或权威渠道。
