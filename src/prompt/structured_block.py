"""结构化输出：schema + few-shot 固定在 system 末尾（近因）。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

__all__ = ["compose_structured_system_prompt", "format_output_block"]

_MARKER = "【结构化输出】"


def format_output_block(
    model: type[BaseModel],
    *,
    examples: list[dict[str, Any]] | None = None,
) -> str:
    schema = model.model_json_schema()
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    lines = [
        _MARKER,
        "你必须只输出一个 JSON 对象，符合下列 JSON Schema，不要 markdown _fence，不要其它说明。",
        schema_json,
    ]
    if examples:
        lines.append("示例（content 与 schema 一致，勿照抄业务字段）：")
        for idx, ex in enumerate(examples, start=1):
            lines.append(f"示例{idx}：{json.dumps(ex, ensure_ascii=False)}")
    return "\n".join(lines)


def compose_structured_system_prompt(
    instructions: str,
    model: type[BaseModel],
    *,
    examples: list[dict[str, Any]] | None = None,
) -> str:
    body = (instructions or "").strip()
    block = format_output_block(model, examples=examples)
    if not body:
        return block
    return f"{body}\n\n{block}"
