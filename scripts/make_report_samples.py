"""生成检查报告测试样例（表格 PDF + 预览 PNG），供「上传检查报告并解读」联调。

用法（项目根）:
    uv run python "B2C business/scripts/make_report_samples.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

B2C_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = B2C_ROOT / "data" / "report_samples"

_SIMSUN = Path(r"C:\Windows\Fonts\simsun.ttc")
_SIMHEI = Path(r"C:\Windows\Fonts\simhei.ttf")

# (项目, 结果, 单位, 参考范围, 提示) —— 值与提示保持临床自洽
REPORTS: list[dict[str, Any]] = [
    {
        "file": "01_血常规_女34岁.pdf",
        "title": "血常规检验报告单",
        "hospital": "示例市第一人民医院",
        "meta": [
            ("姓名", "张某某"),
            ("性别", "女"),
            ("年龄", "34岁"),
            ("科室", "内科门诊"),
            ("标本类型", "全血"),
            ("样本号", "20260910CBC0142"),
            ("采集时间", "2026-09-10 08:12"),
            ("报告时间", "2026-09-10 09:05"),
            ("临床诊断", "发热待查"),
        ],
        "rows": [
            ("白细胞计数", "12.4", "10^9/L", "3.5-9.5", "↑"),
            ("中性粒细胞百分比", "82.1", "%", "40.0-75.0", "↑"),
            ("淋巴细胞百分比", "12.3", "%", "20.0-50.0", "↓"),
            ("单核细胞百分比", "4.6", "%", "3.0-10.0", ""),
            ("嗜酸性粒细胞百分比", "0.8", "%", "0.4-8.0", ""),
            ("嗜碱性粒细胞百分比", "0.2", "%", "0.0-1.0", ""),
            ("中性粒细胞绝对值", "10.2", "10^9/L", "1.8-6.3", "↑"),
            ("淋巴细胞绝对值", "1.53", "10^9/L", "1.1-3.2", ""),
            ("单核细胞绝对值", "0.57", "10^9/L", "0.10-0.60", ""),
            ("嗜酸性粒细胞绝对值", "0.10", "10^9/L", "0.02-0.52", ""),
            ("嗜碱性粒细胞绝对值", "0.02", "10^9/L", "0.00-0.06", ""),
            ("红细胞计数", "3.42", "10^12/L", "3.80-5.10", "↓"),
            ("血红蛋白", "98", "g/L", "115-150", "↓"),
            ("红细胞压积", "31.2", "%", "35.0-45.0", "↓"),
            ("平均红细胞体积", "91.2", "fL", "82.0-100.0", ""),
            ("平均红细胞血红蛋白量", "28.7", "pg", "27.0-34.0", ""),
            ("平均红细胞血红蛋白浓度", "314", "g/L", "316-354", "↓"),
            ("红细胞分布宽度CV", "13.1", "%", "11.5-14.5", ""),
            ("红细胞分布宽度SD", "44.8", "fL", "37.0-54.0", ""),
            ("血小板计数", "428", "10^9/L", "125-350", "↑"),
            ("平均血小板体积", "9.8", "fL", "7.6-13.2", ""),
            ("血小板分布宽度", "16.2", "fL", "9.0-17.0", ""),
            ("血小板压积", "0.42", "%", "0.11-0.28", "↑"),
        ],
    },
    {
        "file": "02_生化全套_男52岁.pdf",
        "title": "生化检验报告单",
        "hospital": "示例市第一人民医院",
        "meta": [
            ("姓名", "李某某"),
            ("性别", "男"),
            ("年龄", "52岁"),
            ("科室", "内分泌科"),
            ("标本类型", "血清"),
            ("样本号", "20260910BIO0388"),
            ("采集时间", "2026-09-10 07:45"),
            ("报告时间", "2026-09-10 11:20"),
            ("临床诊断", "血脂异常"),
        ],
        "rows": [
            ("丙氨酸氨基转移酶", "68", "U/L", "7-40", "↑"),
            ("天门冬氨酸氨基转移酶", "52", "U/L", "13-35", "↑"),
            ("总胆红素", "15.2", "μmol/L", "5.0-21.0", ""),
            ("直接胆红素", "4.1", "μmol/L", "0.0-6.8", ""),
            ("总蛋白", "71.2", "g/L", "65.0-85.0", ""),
            ("白蛋白", "43.5", "g/L", "40.0-55.0", ""),
            ("球蛋白", "27.7", "g/L", "20.0-35.0", ""),
            ("白球比", "1.57", "", "1.20-2.40", ""),
            ("碱性磷酸酶", "88", "U/L", "35-100", ""),
            ("γ-谷氨酰转移酶", "96", "U/L", "7-45", "↑"),
            ("尿素", "5.12", "mmol/L", "2.90-8.20", ""),
            ("肌酐", "78", "μmol/L", "41-81", ""),
            ("尿酸", "386", "μmol/L", "155-357", "↑"),
            ("总胆固醇", "6.31", "mmol/L", "小于5.18", "↑"),
            ("甘油三酯", "2.86", "mmol/L", "小于1.70", "↑"),
            ("高密度脂蛋白胆固醇", "0.92", "mmol/L", "大于1.04", "↓"),
            ("低密度脂蛋白胆固醇", "4.15", "mmol/L", "小于3.37", "↑"),
            ("空腹血糖", "6.84", "mmol/L", "3.90-6.10", "↑"),
            ("糖化血红蛋白", "6.5", "%", "4.0-6.0", "↑"),
        ],
    },
    {
        "file": "03_尿常规_女28岁.pdf",
        "title": "尿常规检验报告单",
        "hospital": "示例市第一人民医院",
        "meta": [
            ("姓名", "王某某"),
            ("性别", "女"),
            ("年龄", "28岁"),
            ("科室", "泌尿外科"),
            ("标本类型", "中段尿"),
            ("样本号", "20260910URI0217"),
            ("采集时间", "2026-09-10 09:30"),
            ("报告时间", "2026-09-10 10:15"),
            ("临床诊断", "尿频尿急"),
        ],
        "rows": [
            ("尿颜色", "淡黄色", "", "淡黄色", ""),
            ("尿透明度", "清晰", "", "清晰", ""),
            ("尿比重", "1.025", "", "1.003-1.030", ""),
            ("尿酸碱度", "6.0", "", "4.5-8.0", ""),
            ("尿蛋白", "1+", "", "阴性", "↑"),
            ("尿葡萄糖", "阴性", "", "阴性", ""),
            ("尿酮体", "阴性", "", "阴性", ""),
            ("尿隐血", "阴性", "", "阴性", ""),
            ("尿胆红素", "阴性", "", "阴性", ""),
            ("尿胆原", "正常", "", "阴性或弱阳性", ""),
            ("亚硝酸盐", "阴性", "", "阴性", ""),
            ("尿白细胞", "2+", "", "阴性", "↑"),
            ("尿红细胞", "3", "/HPF", "0-3", ""),
            ("尿白细胞计数", "18", "/HPF", "0-5", "↑"),
            ("上皮细胞", "少量", "/HPF", "0-5", ""),
            ("管型", "未见", "/LPF", "0-1", ""),
        ],
    },
    {
        "file": "04_甲状腺功能_女45岁.pdf",
        "title": "甲状腺功能检验报告单",
        "hospital": "示例市第一人民医院",
        "meta": [
            ("姓名", "赵某某"),
            ("性别", "女"),
            ("年龄", "45岁"),
            ("科室", "内分泌科"),
            ("标本类型", "血清"),
            ("样本号", "20260910THY0093"),
            ("采集时间", "2026-09-10 08:05"),
            ("报告时间", "2026-09-10 14:40"),
            ("临床诊断", "乏力怕冷"),
        ],
        "rows": [
            ("促甲状腺激素", "8.62", "mIU/L", "0.27-4.20", "↑"),
            ("游离甲状腺素", "10.1", "pmol/L", "12.0-22.0", "↓"),
            ("游离三碘甲状腺原氨酸", "3.85", "pmol/L", "3.10-6.80", ""),
            ("甲状腺过氧化物酶抗体", "612.5", "IU/mL", "0.0-34.0", "↑"),
            ("甲状腺球蛋白抗体", "388.2", "IU/mL", "0.0-115.0", "↑"),
            ("促甲状腺素受体抗体", "0.8", "IU/L", "0.0-1.75", ""),
        ],
    },
]

_DISCLAIMER = "本结果仅对送检标本负责，请结合临床。"
_FOOTER = "检验者：孙某某    审核者：周某某"


def _register_pdf_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for name, path, index in (("SimSun", _SIMSUN, 0), ("SimHei", _SIMHEI, None)):
        if not path.is_file():
            continue
        try:
            font = TTFont(name, str(path)) if index is None else TTFont(name, str(path), subfontIndex=index)
            pdfmetrics.registerFont(font)
        except Exception:
            continue
    names = set(pdfmetrics.getRegisteredFontNames())
    return ("SimSun" if "SimSun" in names else "Helvetica"), ("SimHei" if "SimHei" in names else "Helvetica-Bold")


def _write_pdf(report: dict[str, Any], out_path: Path, *, body: str, head: str) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    title_style = ParagraphStyle("title", fontName=head, fontSize=17, leading=22, alignment=1)
    sub_style = ParagraphStyle("sub", fontName=body, fontSize=10, leading=14, alignment=1)
    note_style = ParagraphStyle("note", fontName=body, fontSize=8.5, leading=12, alignment=0)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=report["title"],
    )

    meta = list(report["meta"])
    while len(meta) % 3:
        meta.append(("", ""))
    meta_rows = [[f"{k}：{v}" if k else "" for k, v in meta[i : i + 3]] for i in range(0, len(meta), 3)]
    meta_table = Table(meta_rows, colWidths=[56 * mm, 56 * mm, 58 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), body),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    data = [["项目", "结果", "单位", "参考范围", "提示"]] + [list(row) for row in report["rows"]]
    style = [
        ("FONTNAME", (0, 0), (-1, -1), body),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("FONTNAME", (0, 0), (-1, 0), head),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for i, row in enumerate(report["rows"], start=1):
        if row[4] == "↑":
            style.append(("TEXTCOLOR", (4, i), (4, i), colors.red))
        elif row[4] == "↓":
            style.append(("TEXTCOLOR", (4, i), (4, i), colors.blue))
    table = Table(data, colWidths=[62 * mm, 22 * mm, 26 * mm, 40 * mm, 20 * mm], repeatRows=1)
    table.setStyle(TableStyle(style))

    story = [
        Paragraph(report["hospital"], sub_style),
        Paragraph(report["title"], title_style),
        Spacer(1, 4 * mm),
        meta_table,
        Spacer(1, 2 * mm),
        table,
        Spacer(1, 3 * mm),
        Paragraph(_DISCLAIMER, note_style),
        Paragraph(_FOOTER, note_style),
    ]
    doc.build(story)


def _write_image(report: dict[str, Any], out_path: Path, *, as_pdf: bool = False) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1754
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    body_font = ImageFont.truetype(str(_SIMSUN), 20, index=0)
    title_font = ImageFont.truetype(str(_SIMHEI), 32)
    head_font = ImageFont.truetype(str(_SIMHEI), 20)
    margin = 90
    y = margin

    draw.text((width / 2, y), report["hospital"], font=head_font, fill="black", anchor="ma")
    y += 40
    draw.text((width / 2, y), report["title"], font=title_font, fill="black", anchor="ma")
    y += 54

    for label, value in report["meta"]:
        draw.text((margin, y), f"{label}：{value}", font=body_font, fill="black")
        y += 30
    y += 10

    cols = [0.36, 0.14, 0.16, 0.24, 0.10]
    table_width = width - 2 * margin
    xs = [margin]
    for frac in cols:
        xs.append(xs[-1] + frac * table_width)
    row_h = 32

    draw.rectangle([margin, y, margin + table_width, y + row_h], fill="#E8E8E8", outline="black")
    for idx, name in enumerate(("项目", "结果", "单位", "参考范围", "提示")):
        draw.text(((xs[idx] + xs[idx + 1]) / 2, y + row_h / 2), name, font=head_font, fill="black", anchor="mm")
    y += row_h

    for row in report["rows"]:
        draw.rectangle([margin, y, margin + table_width, y + row_h], outline="black")
        for idx in range(1, 5):
            draw.line([xs[idx], y, xs[idx], y + row_h], fill="black")
        for idx, cell in enumerate(row):
            fill = "black"
            if idx == 4 and cell == "↑":
                fill = "#C00000"
            elif idx == 4 and cell == "↓":
                fill = "#0040C0"
            anchor = "lm" if idx == 0 else "mm"
            x = xs[idx] + 8 if idx == 0 else (xs[idx] + xs[idx + 1]) / 2
            draw.text((x, y + row_h / 2), str(cell), font=body_font, fill=fill, anchor=anchor)
        y += row_h

    y += 16
    draw.text((margin, y), _DISCLAIMER, font=body_font, fill="black")
    y += 30
    draw.text((margin, y), _FOOTER, font=body_font, fill="black")

    if as_pdf:
        image.save(str(out_path), "PDF", resolution=150.0)
    else:
        image.save(str(out_path), "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成检查报告测试样例")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    body, head = _register_pdf_fonts()

    for report in REPORTS:
        pdf_path = out_dir / report["file"]
        _write_pdf(report, pdf_path, body=body, head=head)
        _write_image(report, out_dir / f"{pdf_path.stem}_preview.png")
        print(f"{report['file']}  项目数={len(report['rows'])}")

    scanned = out_dir / "05_血常规_图片扫描版.pdf"
    _write_image(REPORTS[0], scanned, as_pdf=True)
    print(f"{scanned.name}  无文字层（测 OCR 回退）")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
