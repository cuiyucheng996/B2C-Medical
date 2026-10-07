---
handler: kg_symptom_care
major: 公共知识问答
intent: 查询某症状的生活注意事项
---

## 目标
症状的护理、饮食、运动、现场处理等注意事项。

## 执行步骤
1. 用 search_exam_handbook；只用 data.items / summary。

## 子句拆分（RAGFlow / 手册 / 向量库）
{{include:_snippets/rag_decompose.md}}

## 禁止
- 不要标成未细分急症；明显救命场景应走 risk_emergency。
