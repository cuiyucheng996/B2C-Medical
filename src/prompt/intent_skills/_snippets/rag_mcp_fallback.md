**本地检索不够时（本轮还绑了 MCP 才可用）**

顺序（仍受本轮工具最多 **2 次**限制，证据够就停）：
1. 已按 Skill 调过 `search_exam_handbook`，且 envelope 里 `usable=false` 或 `data.items`/`summary` 仍不够 → 再调 `search_chroma_knowledge`（若本路已绑定）。
2. 手册 + Chroma 仍不够、且需要公开网页科普 → 优先 `web_search`（通义联网）；仍不够可 `brave_web_search`（Brave，query 用名词短语）。
3. 需要运营/合规 FAQ、百炼索引内容 → `bailian_retrieve`（需配置百炼）。
4. 只要抓某一公开 URL 正文 → `fetch`（不要当诊断依据）。

**禁止**：未调本地检索就直接联网；把联网/网页结果当诊断；`add_memory`、清图。
