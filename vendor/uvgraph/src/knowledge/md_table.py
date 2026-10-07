"""通用 Markdown 表格解析（与具体业务字段无关）。"""

#把任意 .md 文件里的 markdown 表格，读成「一行一个字典」的 Python 数据。
# 按行读文件
# 只处理以 | 开头的行（表格行），别的文字跳过
# 第一行有效表格 → 当作表头，记下列名：["疾病", "症状类型", ...]
# | --- | --- | 这种分隔行 → 跳过（_is_separator_row）
# 后面的每一行 → 按 | 拆成格子，和表头配对成字典
# 列数对不齐 → 少的补空字符串，多的截掉
# 整行都是空的 → 不要
from __future__ import annotations

from pathlib import Path


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    """解析 markdown 表格为「表头 -> 单元格」的字典列表。

    适用于任意列名的 .md 表；业务层再按列名映射到自己的 Record。
    """
    if not path.is_file():
        msg = f"markdown table not found: {path}"
        raise FileNotFoundError(msg)

    headers: list[str] | None = None
    rows: list[dict[str, str]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(not cell for cell in cells):
            continue

        if _is_separator_row(cells):
            continue

        if headers is None:
            headers = cells
            continue

        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        elif len(cells) > len(headers):
            cells = cells[: len(headers)]

        row = dict(zip(headers, cells, strict=True))
        if any(value for value in row.values()):
            rows.append(row)

    if headers is None:
        msg = f"no markdown table header found in {path}"
        raise ValueError(msg)
    if not rows:
        msg = f"no markdown table rows parsed from {path}"
        raise ValueError(msg)

    return rows


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(cell.replace(":", "")) <= {"-"} for cell in cells if cell)
