"""把检查手册 PDF 灌进 RAGFlow 知识库（同名覆盖 = 动态更新）。

默认：主项目 DeepDoc 版面识别（title/text）后按 title 分块写入 RAGFlow（不开 TSR）。
RAGFlow 需已启动，并在 Web 创建 API Key 写入 .env 的 RAGFLOW_API_KEY。

用法（项目根）:
    uv run python "B2C business/scripts/ingest_exam_pdf_ragflow.py"
    uv run python "B2C business/scripts/ingest_exam_pdf_ragflow.py" --pdf "路径.pdf"
    uv run python "B2C business/scripts/ingest_exam_pdf_ragflow.py" --tsr
    uv run python "B2C business/scripts/ingest_exam_pdf_ragflow.py" --ragflow-parse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
UVGRAPH_ROOT = B2C_ROOT.parent
B2C_SRC = B2C_ROOT / "src"
DEFAULT_PDF = B2C_ROOT / "document" / "明明白白做医学检查-汪太松-2024(1).pdf"

for path in (str(UVGRAPH_ROOT), str(B2C_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGFlow 灌入/替换公共知识 PDF")
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="PDF 路径")
    parser.add_argument("--keep", action="store_true", help="同名文件不删除，追加一份")
    parser.add_argument("--tsr", action="store_true", help="本地解析时开表格结构识别")
    parser.add_argument("--ragflow-parse", action="store_true", help="改回让 RAGFlow 自己切块")
    args = parser.parse_args()
    from algorithm.ragflow_kb import replace_pdf

    result = replace_pdf(
        args.pdf,
        replace_same_name=not args.keep,
        chunk_source="ragflow" if args.ragflow_parse else "local",
        with_tsr=args.tsr,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    local = result.get("local") or {}
    if local.get("store"):
        print(f"本地分块: {local['store']} 块数={local.get('chunk_count')}")
    remote = result.get("ragflow") or {}
    if remote.get("chunk_count") is not None:
        print(f"RAGFlow 写入块数: {remote.get('chunk_count')} source={remote.get('chunk_source')}")
    if remote.get("web_url"):
        print(f"RAGFlow Web: {remote['web_url']}")


if __name__ == "__main__":
    main()
