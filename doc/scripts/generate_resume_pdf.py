"""生成 doc/崔宇程的简历.pdf（需 Windows 字体 msyh.ttc）。"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "doc" / "崔宇程的简历.pdf"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def _add_paragraph(pdf: FPDF, text: str, size: int = 10, bold: bool = False) -> None:
    style = "B" if bold else ""
    pdf.set_font("msyh", style=style, size=size)
    pdf.multi_cell(0, 5.5, text)
    pdf.ln(1)


def main() -> None:
    if not FONT.is_file():
        raise SystemExit(f"未找到字体: {FONT}")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("msyh", "", str(FONT))
    pdf.add_font("msyh", "B", str(FONT))

    _add_paragraph(pdf, "崔宇程", size=16, bold=True)
    _add_paragraph(
        pdf,
        "求职意向：AI 应用开发 / AI 产品助理 / 大模型应用测试  |  深圳  男  24岁",
        size=10,
    )
    _add_paragraph(
        pdf,
        "13267042550  2033724580@qq.com  共青团员  "
        "https://github.com/cuiyucheng996/B2C-Medical.git",
        size=9,
    )
    pdf.ln(2)

    _add_paragraph(pdf, "自我评价", size=11, bold=True)
    for line in [
        "做过医疗 C 端大模型陪聊全链路：Python、LangGraph、Redis、Neo4j，按环节选组件，不把全栈堆进一次调用。",
        "Agent 按「有限工具、有限轮次」使用：分意图绑工具与 Skill，结果固定结构（可用/报错/摘要/证据），"
        "缺信息或敏感工具走 HITL，不让模型猜。",
        "习惯对着 Badcase 与失败请求改 Prompt、阈值和降级策略；医疗场景强调非诊断、证据溯源。",
        "熟悉 Prompt/Skill 维护、RAGAS 评测脚本、FastAPI+SSE；了解 Dify/Coze/n8n 搭原型；"
        "有工业视觉与舌诊图像检测联调经验。",
    ]:
        _add_paragraph(pdf, "· " + line, size=9)

    pdf.ln(2)
    _add_paragraph(pdf, "项目经历", size=11, bold=True)
    _add_paragraph(
        pdf,
        "医疗 C 端健康陪聊 Agent（LangGraph + LangChain）  独立设计  2026.7 — 2026.10",
        size=10,
        bold=True,
    )
    _add_paragraph(pdf, "仓库：https://github.com/cuiyucheng996/B2C-Medical.git", size=9)
    _add_paragraph(
        pdf,
        "简介：从对话与上传材料采集病理实体并沉淀长期记忆；多意图并行编排（知识/报告/记忆/提醒），"
        "高风险拦截，非诊断、不开药。",
        size=9,
    )
    for line in [
        "预处理：文本纠错与指代补全；多格式文件解析与化验单规则打分（≥2 分走报告解读），下游统一结构化输入。",
        "意图：FAQ 直出降链路；否则 LLM 结构化多意图（可配置阈值），失败降级规则/BERT，输出多路子任务。",
        "编排：LangGraph 分发 → Send 并行子 Agent → 出口合并；单/多任务 SSE 流式推送。",
        "子 Agent：分意图 create_agent + 专属工具/MCP/Skill，ReAct 多轮；工具结果统一封装；System/Skill 前缀缓存降 Token。",
        "RAG：实体对齐后查 Neo4j 图谱与向量/RAGFlow；多路召回 RRF + 关键词精排；Skill 指导拆句做 Agentic RAG。",
        "HITL：interrupt + checkpoint 续跑（工具确认、实体补槽、提醒缺时间、报告缺文件）；关场写 Finding 入 Neo4j。",
    ]:
        _add_paragraph(pdf, "· " + line, size=9)

    _add_paragraph(pdf, "技术栈：Python 3.11、FastAPI、SSE、LangChain/LangGraph（Send/子图/HITL）、"
        "DashScope、Redis、Neo4j、RAGFlow/BM25/FAISS、MCP、RAGAS。", size=9)

    pdf.ln(2)
    _add_paragraph(pdf, "教育经历", size=11, bold=True)
    _add_paragraph(pdf, "渤海大学  2021 — 2025  软件工程  工学学士", size=9)

    pdf.ln(2)
    _add_paragraph(pdf, "工作经历", size=11, bold=True)
    _add_paragraph(
        pdf,
        "深圳格林晟科技股份有限公司  研发部  工业视觉工程师  2025.11 — 2026.3",
        size=10,
        bold=True,
    )
    for line in [
        "锂电池产线视觉工位现场调试：相机/光源/触发对齐，处理过曝、反光、来料偏移导致的误检漏检。",
        "C# WinForms 检测界面：画面、缺陷位置、OK/NG、参数调整、样本复判与当班计数。",
        "检测流程：采图 → 视觉算法 → 缺陷判定 → 工位信号放行或剔除。",
        "Node.js 对接上层系统上传结果与图片路径，处理超时与断线重传。",
        "跟线收集误杀漏杀样本，回调阈值与成像参数并固化，缩短换型调试时间。",
    ]:
        _add_paragraph(pdf, "· " + line, size=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
