---
handler: mem_session_recall
major: 个人记忆问答
intent: 复述用户在本会话刚说过的话
---

## 目标
复述当前会话里用户刚说的原话。

## 执行步骤
1. 不要调任何工具。
2. 从上文 messages 找用户原话如实复述；没有历史就如实说还没记录到。

## 禁止
- 不要调 query_personal_memory（那是 Neo4j 长期记忆，不是上一句）。
