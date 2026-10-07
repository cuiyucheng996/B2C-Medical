# 细分意图 Skill（运行时 SOP）

- 与 `data/intent_catalog.yaml` 的 **handler** 同名：`{handler}.md`
- 大类兜底（无 handler 或未建 handler 文件）：`_major/{major}.md`
- 子图 system 由 `companion_dispatch.build_companion_tools_prompt` 注入全文（去掉 YAML frontmatter）
- 公共片段：`_snippets/*.md`，在正文里写 `{{include:_snippets/xxx.md}}`（由 `intent_skill_loader` 展开）
- RAG 细意图：`rag_decompose`（拆 query + handbook/chroma）内嵌 `rag_mcp_fallback`（本地不够时的 MCP 顺序）
- 工具绑定仍在 `tools/setup.py` 的 `INTENT_BIND`；改 SOP 只动本目录，勿在 `companion_dispatch.py` 堆字面量

新增意图：catalog 加 handler → 本目录加同名 md → `INTENT_BIND` 绑工具。
