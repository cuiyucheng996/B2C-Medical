---
handler: mem_stated_consult
major: 个人记忆问答
intent: 查询用户曾经提到的部位或疾病
---

## 目标
回答用户曾提到的部位、疾病、症状（画像 + Finding）。

## 执行步骤
1. 只调 lookup_stated_consult 一次。
2. 用 usable 的 extracts / current_* 组句。

## 禁止
- 不要调 lookup_life_facts。
