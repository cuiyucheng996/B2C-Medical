# -*- coding: utf-8 -*-
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).resolve().parent
INK = HexColor("#1c241e")
MUTED = HexColor("#5c6b62")
ACCENT = HexColor("#2f6f4e")
LINE = HexColor("#d5e0d8")


def _font() -> str:
    for path, name in (
        (r"C:\Windows\Fonts\msyh.ttc", "YaHei"),
        (r"C:\Windows\Fonts\msyh.ttf", "YaHei"),
        (r"C:\Windows\Fonts\simhei.ttf", "SimHei"),
        (r"C:\Windows\Fonts\simsun.ttc", "SimSun"),
    ):
        if Path(path).is_file():
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=0))
            return name
    return "Helvetica"


FONT = _font()


def styles():
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "name", parent=base["Title"], fontName=FONT, fontSize=20, textColor=INK, alignment=TA_CENTER, spaceAfter=2 * mm
        ),
        "role": ParagraphStyle(
            "role", parent=base["Normal"], fontName=FONT, fontSize=12, textColor=ACCENT, alignment=TA_CENTER, spaceAfter=3 * mm
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["Normal"], fontName=FONT, fontSize=9, textColor=MUTED, alignment=TA_CENTER, spaceAfter=5 * mm
        ),
        "h": ParagraphStyle(
            "h", parent=base["Heading2"], fontName=FONT, fontSize=11, textColor=ACCENT, spaceBefore=4 * mm, spaceAfter=2 * mm
        ),
        "p": ParagraphStyle(
            "p", parent=base["Normal"], fontName=FONT, fontSize=9.5, textColor=INK, leading=15, alignment=TA_JUSTIFY, spaceAfter=2 * mm
        ),
        "b": ParagraphStyle(
            "b", parent=base["Normal"], fontName=FONT, fontSize=9.5, textColor=INK, leading=14.5, leftIndent=2 * mm, spaceAfter=1 * mm
        ),
        "sub": ParagraphStyle(
            "sub", parent=base["Normal"], fontName=FONT, fontSize=9.5, textColor=INK, leading=14, spaceAfter=1 * mm
        ),
    }


def bullets(items: list[str], st) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, st["b"]), leftIndent=8, bulletColor=ACCENT) for item in items],
        bulletType="bullet",
        start="•",
        leftIndent=10,
        bulletFontName=FONT,
        bulletFontSize=9,
    )


def header(story, st, title: str, subtitle: str) -> None:
    story.append(Paragraph("（请替换）姓名", st["name"]))
    story.append(Paragraph(title, st["role"]))
    story.append(Paragraph(subtitle, st["meta"]))
    story.append(
        Paragraph(
            "电话 / 邮箱 / 城市请自行替换　　Git：uvgraph（中医问诊图 + C 端陪聊 Agent）",
            st["meta"],
        )
    )


def build_agent(path: Path) -> None:
    st = styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="简历-Agent应用开发",
    )
    story = []
    header(
        story,
        st,
        "应聘：Agent 应用开发 / LLM 应用工程师",
        "方向：LangGraph 状态机 · 工具编排 · 意图与槽位 · 评测闭环",
    )
    story.append(Paragraph("求职摘要", st["h"]))
    story.append(
        Paragraph(
            "面向医疗问诊与 C 端陪聊，用 LangGraph / LangChain Agent 把「意图 → 抽槽对齐 → 工具召回 → 人机协同」做成可中断的状态机，"
            "而不是单轮 Chat。熟悉工具绑定、结构化输出、SSE 流式、HITL 追问，以及用评测台把细意图、图谱召回、上传闸门从业务路径里复刻出来验收。",
            st["p"],
        )
    )
    story.append(Paragraph("核心能力", st["h"]))
    story.append(
        bullets(
            [
                "Agent 编排：LangGraph 共享状态、条件路由、interrupt；LangChain create_agent + ToolStrategy，按细意图挂不同工具。",
                "意图与槽位：大类锁定后 LLM 限时选细意图，失败降级规则；UIE 抽实体 + 同义词表 / Milvus 粗召回 + 精排对齐到 Neo4j 标准名。",
                "工具与证据：图谱只走 Cypher（部位-病、病-症、证型）；检查手册走 RAGFlow；禁止模型编清单。",
                "上传链路：YOLOX 版面打分锁「报告指标解读」，可转 PDF 则 MinerU 拆表抽指标，再对照手册。",
                "工程：FastAPI + SSE、评测 HTML、HITL、Graphiti 长期记忆、n8n 任务、语音 ASR/TTS；配置与密钥单一入口。",
            ],
            st,
        )
    )
    story.append(Paragraph("项目一　C 端陪聊 Agent（B2C business）", st["h"]))
    story.append(
        Paragraph(
            "角色：应用开发　　栈：Python 3.11 / FastAPI / LangChain Agent / DeepSeek / Neo4j / RAGFlow / Milvus / MinerU",
            st["sub"],
        )
    )
    story.append(
        bullets(
            [
                "陪聊主路径：预处理（纠错、指代、意图）后按大类绑工具；闲聊不挂业务工具，公共知识按细意图只调图谱或手册。",
                "细意图 8 类（部位常见病、病-症、症-部位、通俗解释、检查含义、指标范围、症状护理、证型表现）与 handler、工具表一一对应。",
                "query_consult_graph 只消费进工具前的对齐结果跑固定 Cypher；评测复刻「细意图 → 抽实体对齐 → 工具」全链路。",
                "文件闸门：表区域 / 化验词 / 版面失败加分，≥2 分锁报告大类；图片转 PDF 后交给 explain_uploaded_report。",
                "线上问题：修复发送按钮把 PointerEvent 当文案；评测意图卡死改为 LLM 5 秒硬超时；工具消息与 tool_calls 不对齐导致 400 的排查。",
            ],
            st,
        )
    )
    story.append(Paragraph("项目二　B 端问诊状态机（uvgraph / B2B）", st["h"]))
    story.append(
        Paragraph(
            "角色：应用开发　　栈：LangGraph / 部位+疾病锁定 / UIE / entity_align / Neo4j Consult* / 证型匹配",
            st["sub"],
        )
    )
    story.append(
        bullets(
            [
                "第一大轮锁定部位与疾病类型：选项快路径与文字链（澄清、相关性、解析、匹配）用路由函数看状态字段，不用 docstring 驱动。",
                "抽槽对齐权威实现：同义词表 → 精确名 → Milvus + 精排；C 端对齐内核复用该模块，编排按陪聊多意图展开。",
                "图 schema：ConsultBodyPart / ConsultDisease / 主症 / 证型，BELONG 关系；口语「头部」对齐「头面部」后再入图。",
                "节点保持薄：算法在 algorithm/，图节点只读写 RouterState；业务字面量集中 schemas，配置集中 .env + Settings。",
            ],
            st,
        )
    )
    story.append(Paragraph("技术关键词", st["h"]))
    story.append(
        Paragraph(
            "LangGraph · LangChain · Tool Calling · Structured Output · HITL / interrupt · FastAPI SSE · "
            "Neo4j Cypher · RAGFlow · Milvus · UIE · YOLOX / MinerU · Graphiti · DeepSeek / DashScope · uv / pytest",
            st["p"],
        )
    )
    story.append(Paragraph("教育 / 工作（请替换）", st["h"]))
    story.append(Paragraph("学历、在职时间、上一段职责请自行填写。以上项目经历可按实际参与深度删减条目。", st["p"]))
    doc.build(story)


def build_fde(path: Path) -> None:
    st = styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="简历-FDE",
    )
    story = []
    header(
        story,
        st,
        "应聘：Forward Deployed Engineer / 现场交付工程师",
        "方向：把客户场景落成可运行的 Agent 与知识链路，能部署、能排障、能改流程",
    )
    story.append(Paragraph("求职摘要", st["h"]))
    story.append(
        Paragraph(
            "在中医问诊 B 端与陪聊 C 端之间做「能上线的交付」：对接 Neo4j / RAGFlow / Milvus / 语音服务，"
            "把客户说法映射成意图、标准实体和工具结果，而不是只调通一个 Demo Chat。"
            "习惯用评测台和日志把召回空、超时、版面失败变成可复现的验收项，现场改配置与路径即可闭环。",
            st["p"],
        )
    )
    story.append(Paragraph("交付相关能力", st["h"]))
    story.append(
        bullets(
            [
                "场景拆解：公共知识问答 8 细类对应图谱或手册；报告上传用打分闸门分流，避免所有文件都 OCR。",
                "数据与检索：Neo4j 问诊图种子对齐、RAGFlow 手册块、Milvus 实体对齐；会解释「通道对了但召回空」是标准名不一致。",
                "现场排障：RAGFlow 超时、LLM 5 秒意图超时、工具历史 400、单线程 uvicorn 被长请求堵住；能看 access log 与业务 traceback。",
                "验收资产：C 端 /eval 复刻细意图+召回、指代补全、上传打分/PDF/拆表，给客户演示不依赖陪聊全链路。",
                "集成：n8n 任务提醒、ASR/TTS、Graphiti 记忆、文件 15MB 限制与格式嗅探；密钥走 .env，不散落业务代码。",
            ],
            st,
        )
    )
    story.append(Paragraph("项目一　C 端陪聊上线形态（对客户可点）", st["h"]))
    story.append(
        Paragraph("栈：FastAPI · 静态评测页 · SSE · RAGFlow · MinerU · 语音 8000/5050", st["sub"])
    )
    story.append(
        bullets(
            [
                "交付物：登录会话、流式陪聊、上传报告、评测台、工具注册表；改 Python 需重启（无 reload），现场改完可验证。",
                "知识交付：图谱题必须走 Neo4j，检查题走 RAGFlow；评测页展示对齐「头部→头面部」与召回块，避免客户以为系统没查。",
                "报告交付：YOLOX 判表 + 化验词打分；报告类转 PDF 后 MinerU 拆表；非报告图不进解读工具。",
                "客户话术复杂句可落到细意图与槽，用于现场验收而不是只测「你好」。",
            ],
            st,
        )
    )
    story.append(Paragraph("项目二　B 端问诊图与对齐（对医院/运营可解释）", st["h"]))
    story.append(
        Paragraph("栈：LangGraph 问诊轮次 · MySQL 灌图 · 同义词表 · Milvus 对齐", st["sub"])
    )
    story.append(
        bullets(
            [
                "能向业务解释：口语部位必须对齐图上标准名（头面部/颈部/胸胁部等），否则 Cypher 精确匹配为空。",
                "对齐策略可配置讲述：同义词命中最快，向量召回 + 精排兜底；与问诊轮次填槽（一轮部位、二轮疾病）的差异能讲清楚。",
                "图关系 BELONG：部位含病、病含主症/证型；现场查库可用 Cypher 对账，不把 RAG 块误当成图谱结果。",
            ],
            st,
        )
    )
    story.append(Paragraph("驻场工作方式", st["h"]))
    story.append(
        bullets(
            [
                "先对齐意图清单和工具绑定，再谈模型；先打通一条金标路径（部位问病 / 血常规手册 / 一张化验单），再铺开。",
                "配置与缓存不写系统盘习惯（uv 缓存、项目内 venv），便于在客户机复现环境。",
                "文档与界面中文；能把超时、空召回、版面失败翻译成业务语言和改法（改词表、改超时、重启进程、查 Neo4j 节点名）。",
            ],
            st,
        )
    )
    story.append(Paragraph("技术关键词", st["h"]))
    story.append(
        Paragraph(
            "客户现场交付 · 流程落地 · Neo4j · RAGFlow · Milvus · LangGraph · 评测验收 · 日志排障 · "
            "Docker/本地多服务联调 · FastAPI · 医疗知识图谱",
            st["p"],
        )
    )
    story.append(Paragraph("教育 / 工作（请替换）", st["h"]))
    story.append(Paragraph("学历、可出差/驻场意愿、上一段交付项目请自行填写。", st["p"]))
    doc.build(story)


def main() -> None:
    build_agent(OUT / "简历-Agent应用开发.pdf")
    build_fde(OUT / "简历-FDE现场交付.pdf")
    print("ok", OUT)


if __name__ == "__main__":
    main()
