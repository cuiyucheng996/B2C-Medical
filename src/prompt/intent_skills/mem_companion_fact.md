---
handler: mem_companion_fact
major: 个人记忆问答
intent: 查询用户最近的心情或事件
---

## 目标
回答用户最近心情、事件、生活事实（入库 Finding）。

## 执行步骤
1. 只调 lookup_life_facts；query 可填用户关心点。
2. 用 facts / summary 组句。

## 禁止
- 不要调 lookup_stated_consult。
