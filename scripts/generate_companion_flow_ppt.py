"""生成 C 端陪聊流程说明 PPT（医疗风格）。输出到 B2C business/doc/。"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

B2C_ROOT = Path(__file__).resolve().parents[1]
# Unicode 转义，避免 Windows 下源文件编码导致乱码文件名
OUT = B2C_ROOT / "doc" / "C\u7aef\u966a\u804a\u5b8c\u6574\u6d41\u7a0b.pptx"
OUT_ASCII = B2C_ROOT / "doc" / "B2C_companion_full_flow.pptx"

# 医疗风配色
C_PRIMARY = RGBColor(0x00, 0x79, 0x6B)
C_PRIMARY_DARK = RGBColor(0x00, 0x4D, 0x40)
C_ACCENT = RGBColor(0x4D, 0xB6, 0xAC)
C_BG_SOFT = RGBColor(0xF0, 0xF7, 0xF6)
C_TEXT = RGBColor(0x26, 0x32, 0x38)
C_TEXT_MUTED = RGBColor(0x54, 0x6E, 0x7A)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _set_slide_bg(slide, rgb: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb


def _add_header_bar(slide, title: str, *, subtitle: str = "") -> None:
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE.RECTANGLE
        Inches(0),
        Inches(0),
        Inches(13.333),
        Inches(1.15),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_PRIMARY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    p.font.name = "Microsoft YaHei"
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(14)
        p2.font.color.rgb = C_ACCENT
        p2.font.name = "Microsoft YaHei"


def _add_footer(slide, text: str = "B2C 智能陪聊 · uvgraph") -> None:
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12), Inches(0.35))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = C_TEXT_MUTED
    p.font.name = "Microsoft YaHei"
    p.alignment = PP_ALIGN.RIGHT


def _add_bullets(slide, items: list[str], *, top: float = 1.45, width: float = 12.2) -> None:
    box = slide.shapes.add_textbox(Inches(0.65), Inches(top), Inches(width), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    for i, line in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(17) if not line.startswith("  ") else Pt(15)
        p.font.color.rgb = C_TEXT if not line.startswith("  ") else C_TEXT_MUTED
        p.font.name = "Microsoft YaHei"
        p.space_after = Pt(8)
        if line.startswith("•"):
            p.level = 0
        elif line.startswith("  "):
            p.level = 1


def _slide_title_only(prs: Presentation, title: str, subtitle: str, tagline: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, C_BG_SOFT)
    accent = slide.shapes.add_shape(1, Inches(0), Inches(2.8), Inches(13.333), Inches(0.08))
    accent.fill.solid()
    accent.fill.fore_color.rgb = C_ACCENT
    accent.line.fill.background()

    tbox = slide.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.5), Inches(1.5))
    p = tbox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_DARK
    p.font.name = "Microsoft YaHei"

    sbox = slide.shapes.add_textbox(Inches(0.8), Inches(3.0), Inches(11.5), Inches(1.2))
    p2 = sbox.text_frame.paragraphs[0]
    p2.text = subtitle
    p2.font.size = Pt(22)
    p2.font.color.rgb = C_PRIMARY
    p2.font.name = "Microsoft YaHei"

    gbox = slide.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.5), Inches(1.5))
    p3 = gbox.text_frame.paragraphs[0]
    p3.text = tagline
    p3.font.size = Pt(16)
    p3.font.color.rgb = C_TEXT_MUTED
    p3.font.name = "Microsoft YaHei"

    cross = slide.shapes.add_shape(1, Inches(11.2), Inches(0.5), Inches(1.6), Inches(0.22))
    cross.fill.solid()
    cross.fill.fore_color.rgb = C_ACCENT
    cross.line.fill.background()
    cross2 = slide.shapes.add_shape(1, Inches(11.85), Inches(0), Inches(0.22), Inches(1.6))
    cross2.fill.solid()
    cross2.fill.fore_color.rgb = C_ACCENT
    cross2.line.fill.background()

    _add_footer(slide, "FastAPI :8002 · LangGraph · 健康陪聊")


def _slide_section(prs: Presentation, section: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, C_PRIMARY)
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.8), Inches(11.5), Inches(1.5))
    p = box.text_frame.paragraphs[0]
    p.text = section
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    p.font.name = "Microsoft YaHei"
    _add_footer(slide, "")


def _slide_content(prs: Presentation, title: str, bullets: list[str], subtitle: str = "") -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _set_slide_bg(slide, C_WHITE)
    stripe = slide.shapes.add_shape(1, Inches(0), Inches(1.15), Inches(0.12), Inches(6.35))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = C_ACCENT
    stripe.line.fill.background()
    _add_header_bar(slide, title, subtitle=subtitle)
    _add_bullets(slide, bullets)
    _add_footer(slide)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    _slide_title_only(
        prs,
        "C 端智能陪聊",
        "完整业务流程与技术架构",
        "Session + LangGraph 多意图并行 + Agent 工具链 + 医疗知识/memory",
    )

    _slide_content(
        prs,
        "产品定位",
        [
            "• B2C 健康场景陪聊：单轮/多轮对话，支持报告、记忆、知识问答与任务提醒",
            "• 生产入口：B2C business/run.py → FastAPI http://127.0.0.1:8002",
            "• 设计原则：预处理给观察上下文；图谱与长期记忆由意图绑定工具按需查询",
            "• 基础设施：DashScope LLM、Redis 本场记忆、Neo4j 用户与知识、Chroma 向量",
        ],
    )

    _slide_section(prs, "一、业务场景")

    _slide_content(
        prs,
        "八大意图大类（intent_catalog）",
        [
            "• 风险拦截 — 急症、用药剂量等安全拦截",
            "• 报告指标解读 — 上传 PDF/表格、指标对比历史",
            "• 个人记忆问答 — 用户曾说的部位、疾病、生活事实",
            "• 具体任务 — 服药提醒、复查、打卡、预约",
            "• 公共知识问答 — Neo4j 知识图 + 检查手册 + 向量检索",
            "• 近况陈述 — 身体不适、心情事件、图片/文档",
            "• 闲聊 — 寒暄、感谢",
            "• 一句用户话可拆成多条 intent_tasks，LangGraph 并行处理",
        ],
        subtitle="data/intent_catalog.yaml",
    )

    _slide_section(prs, "二、系统架构")

    _slide_content(
        prs,
        "分层架构",
        [
            "• 前端 /chat + 调试台 → POST .../chat/stream（SSE）",
            "• api.server：SessionStore、CompanionSession（session_id / user_id）",
            "• algorithm：预处理、意图、session_memory、对齐与压缩",
            "• graph：LangGraph 主图 + intent_worker 子图 + merge / peer_handoff",
            "• agent：create_agent + 中间件栈 + CompanionReply 结构化输出",
            "• tools：本地 CompanionToolRegistry + MCP Client（config.yaml）",
            "• 存储：Redis（本场 short/compress/slots）、Neo4j（用户记忆/知识图）、Chroma 向量",
        ],
    )

    _slide_content(
        prs,
        "会话与记忆（C 端）",
        [
            "• user_id：用户维度（槽位、Neo4j UserInfo、长期事实）",
            "• session_id：单场对话（Redis short/compress、LangGraph thread_id）",
            "• 关场：持久化 Finding + release_companion_session 释放 Redis/checkpoint",
            "• LangGraph checkpoint 仅用于本轮/HITL 续跑，不是长期记忆仓库",
        ],
        subtitle="session.py · algorithm/session_memory.py",
    )

    _slide_section(prs, "三、一轮对话全链路")

    _slide_content(
        prs,
        "用户视角时序（8 步）",
        [
            "1. 用户消息 → Session 写入，同步 Redis 短期历史",
            "2. run_chat_preprocess：纠错 → 指代 → 意图 → 实体/对齐 → observe 观察串",
            "3. HITL 挂起：缺槽位等 → 返回 question + resume_token",
            "4. _attach_intent：每条意图 → major / intent / handler + 绑定工具名",
            "5. ainvoke_companion_intent_graph：Send 并行或 Supervisor（可配置）",
            "6. intent_worker：Agent 调工具 → CompanionReply 草稿",
            "7. review_companion_draft：评审不过则限次重跑 execute",
            "8. merge_node → reply + trace → SSE state（round.langgraph）→ done",
        ],
    )

    _slide_content(
        prs,
        "预处理流水线",
        [
            "• 纠错：BERT / 纠错 Agent 链（可选）",
            "• 指代补全：结合最近用户话（LLM 或跳过）",
            "• 意图：BERT 大类分 + 阈值；细意图规则 + LLM；可多任务并行",
            "• 实体/对齐：部位病名等 → Chroma 对齐（部分大类跳过）",
            "• format_preprocess_for_agent → Agent 的 observe，不替代工具查库",
        ],
        subtitle="algorithm/chat_preprocess.py",
    )

    _slide_section(prs, "四、LangGraph 编排")

    _slide_content(
        prs,
        "主图（默认 Send 模式）",
        [
            "• START → begin_turn（清空当轮 worker_outputs、handoff 计数）",
            "• dispatch_intents → Send × N → intent_worker",
            "• route_after_intent_worker：单任务 → merge；多任务 → peer_handoff",
            "• peer_handoff：wave1 初稿进 wave1_outputs，二轮带同伴摘要再 execute",
            "• merge：合并终稿 + trace（tool_runs 合并 wave1+wave2 工具 trace）",
            "• 可选 COMPANION_USE_SUPERVISOR=1：Supervisor + transfer_to 子智能体",
        ],
        subtitle="graph/companion.py",
    )

    _slide_content(
        prs,
        "intent_worker 子图",
        [
            "• execute：绑 tools_for_intent + MCP(major) → _ainvoke_companion_agent",
            "• review：检查草稿是否跑题、是否缺证据",
            "• 不通过且未超重试 → 带 review_hint 再打回 execute",
            "• finish → worker_outputs（reply、tool_trace、middleware_trace、评审结果）",
            "• 单 worker 流式吐 CompanionReply；多 worker 由 merge 分块推 token",
        ],
    )

    _slide_section(prs, "五、Agent 与工具")

    _slide_content(
        prs,
        "中间件栈（节选）",
        [
            "• Observe / Prompt — 注入预处理观察与系统提示",
            "• EntityMemory — 实体/图谱片段注入（≠ search_long_memory 工具）",
            "• Compress / Think — 上下文压缩与思考模式",
            "• ToolRetry / ToolError / McpGuard — 重试、错误、MCP 白名单",
            "• ToolEnvelope — 统一 CompanionToolEnvelope 给模型",
            "• ToolTrace — 调试台 tool_runs.steps",
            "• HumanInTheLoop（可选）— 敏感工具人工确认",
        ],
    )

    _slide_content(
        prs,
        "工具与 MCP",
        [
            "• INTENT_BIND（tools/setup.py）：(major, handler) → 本地工具子集",
            "• 示例：query_consult_graph、lookup_life_facts、search_long_memory、报告/任务类",
            "• MCP：mcp_bridge 读 config.yaml，按 major 绑定；stdio / HTTP",
            "• MCP = 连接协议；CompanionToolEnvelope = 模型读结果（ok/usable/data）",
            "• 详见 doc/mcp.md",
        ],
    )

    _slide_content(
        prs,
        "知识与记忆",
        [
            "• 知识图：Neo4j ← query_consult_graph（部位-病-症等）",
            "• 手册/向量：Chroma（Docker 8000）← search_exam_handbook / search_chroma_knowledge",
            "• 用户画像：lookup_stated_consult；生活事实：lookup_life_facts",
            "• 近况：search_long_memory + 实体记忆中间件",
            "• 本场：Redis conversation:short / compress:{session_id}",
            "• 槽位：conversation:slots:{user_id}，关场校验 session_id",
        ],
    )

    _slide_section(prs, "六、可观测与运维")

    _slide_content(
        prs,
        "Trace 与调试",
        [
            "• round.langgraph：reviews、tool_runs（steps + middleware_trace）、pattern",
            "• llm_calls：指代、判意图、绑定工具、工具 Envelope 明细",
            "• checkpoint_debug：释放前快照；默认 retention=run 删 LangGraph thread",
            "• 测流程看 SSE state 里的 trace；释放后勿依赖 get_state 拉 graph_state",
            "• 评测：/api/eval/*（意图、KB、记忆、RAG 配置）",
        ],
    )

    _slide_content(
        prs,
        "配置与启动",
        [
            "• uv run python \"B2C business/run.py\" --port 8002",
            "• .env：DashScope、Redis、Neo4j、Embedding、MCP 相关",
            "• COMPANION_PEER_HANDOFF / COMPANION_USE_SUPERVISOR / CHECKPOINT_RETENTION",
            "• 向量：Chroma + bootstrap_chroma_vectors；Neo4j 关系需与 schema 一致",
            "• GET /healthcheck；改 MCP YAML 后重启 API",
        ],
    )

    _slide_title_only(
        prs,
        "谢谢",
        "演示建议：多意图问句 + 调试台查看 tool_runs",
        "文档：B2C business/doc/mcp.md · scripts/generate_companion_flow_ppt.py",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_ASCII))
    try:
        prs.save(str(OUT))
    except OSError:
        pass
    return OUT_ASCII


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
