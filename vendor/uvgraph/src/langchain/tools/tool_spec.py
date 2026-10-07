"""ToolSpec：工具说明书（路由质量的单一权威源）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

__all__ = [
    "ToolSpec",
    "clean_json_schema",
    "spec_to_langchain_tool",
]


def clean_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """清理 JSON Schema，兼容 OpenAI / DeepSeek tools 绑定。"""
    if not schema:
        return {"type": "object", "properties": {}}

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in {"$defs", "definitions"}:
            cleaned[key] = _clean_defs(value) if isinstance(value, dict) else value
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop: clean_json_schema(field) if isinstance(field, dict) else field
                for prop, field in value.items()
            }
            continue
        if key == "items" and isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
            continue
        if key in {"title", "description"} and key == "title":
            continue
        cleaned[key] = value

    cleaned.pop("title", None)
    if cleaned.get("type") == "object" and "additionalProperties" not in cleaned:
        cleaned["additionalProperties"] = False
    return cleaned


def _clean_defs(defs: dict[str, Any]) -> dict[str, Any]:
    return {name: clean_json_schema(defn) if isinstance(defn, dict) else defn for name, defn in defs.items()}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """工具规格：name + description + parameters → OpenAI tools JSON。"""

    name: str
    description: str
    parameters: type[BaseModel]
    needs_confirmation: bool = False
    visible_to: str = "main"

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI tools 格式（清理 DeepSeek 不兼容字段）。"""
        schema = clean_json_schema(self.parameters.model_json_schema())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": schema,
            },
        }

    def to_detail_dict(self) -> dict[str, Any]:
        schema = self.parameters.model_json_schema()
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        return {
            "name": self.name,
            "description": self.description.strip(),
            "parameters_schema": schema,
            "parameters": [
                {
                    "name": key,
                    "type": str(field.get("type", "string")),
                    "required": key in required,
                    "description": str(field.get("description") or ""),
                    "enum": field.get("enum"),
                }
                for key, field in properties.items()
            ],
            "needs_confirmation": self.needs_confirmation,
            "visible_to": self.visible_to,
            "openai_schema": self.to_openai_schema(),
        }


def spec_to_langchain_tool(
    spec: ToolSpec,
    *,
    fallback: BaseTool | None = None,
    invoke: Callable[..., Any] | None = None,
) -> BaseTool:
    """由 ToolSpec 构造或绑定 LangChain Tool。"""
    if fallback is not None:
        return fallback
    fn = invoke or spec.invoke
    if fn is None:
        raise ValueError(f"ToolSpec {spec.name!r} 缺少 invoke 实现")
    return StructuredTool.from_function(
        func=fn,
        name=spec.name,
        description=spec.description.strip(),
        args_schema=spec.parameters,
    )
