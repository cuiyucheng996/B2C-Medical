---
handler: risk_emergency
major: 风险拦截
intent: 拦截急症或求救描述
---

## 目标
急症/求救：引导就医，不给自行处置方案。

## 执行步骤
1. 只调 safety_policy_hint。
2. 组句强调尽快线下就医/急救，勿替代 120。

## 禁止
- 不要下诊断、不要给具体用药剂量。
