"""Agent 结构化输出：API 约束 + Pydantic 校验 + 校验失败回灌自修复 + 可选降级。"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from algorithm.json_sanitize import extract_json_object_text, strip_markdown_json_fence

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

__all__ = [
    "invoke_structured_llm",
    "parse_json_with_schema",
    "structured_methods",
    "structured_max_repair",
]

_FALLBACK = Callable[[], SchemaT]


def structured_max_repair() -> int:
    raw = (os.getenv("COMPANION_STRUCTURED_MAX_REPAIR") or "3").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(0, min(n, 5))


def structured_methods() -> tuple[str, ...]:
    raw = (os.getenv("COMPANION_STRUCTURED_METHODS") or "function_calling,json_mode").strip()
    order = [p.strip() for p in raw.split(",") if p.strip()]
    allowed = {"json_schema", "function_calling", "json_mode", "json_mode_strict"}
    out = [m for m in order if m in allowed]
    return tuple(out or ("function_calling", "json_mode"))


def _coerce_model(schema: type[SchemaT], payload: Any) -> SchemaT:
    if isinstance(payload, schema):
        return payload
    if isinstance(payload, dict):
        return schema.model_validate(payload)
    if isinstance(payload, str):
        return parse_json_with_schema(payload, schema)
    raise ValidationError.from_exception_data(
        schema.__name__,
        [{"type": "model_type", "loc": (), "input": payload, "msg": "非 dict/str"}],
    )


def parse_json_with_schema(text: str, schema: type[SchemaT]) -> SchemaT:
    blob = extract_json_object_text(text)
    if not blob:
        raise ValidationError.from_exception_data(
            schema.__name__,
            [{"type": "json_invalid", "loc": (), "input": text, "msg": "未找到 JSON 对象"}],
        )
    try:
        row = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValidationError.from_exception_data(
            schema.__name__,
            [{"type": "json_invalid", "loc": (), "input": strip_markdown_json_fence(text), "msg": str(exc)}],
        ) from exc
    return schema.model_validate(row)


def _invoke_once(model: Any, schema: type[SchemaT], messages: list[BaseMessage], method: str) -> SchemaT:
    structured_model = model.with_structured_output(schema, method=method)
    analysis = structured_model.invoke(messages)
    return _coerce_model(schema, analysis)


def _escalate_model(model: Any) -> Any:
    """COMPANION_STRUCTURED_ESCALATE_MODEL 非空时再试一次更强模型（默认关）。"""
    name = (os.getenv("COMPANION_STRUCTURED_ESCALATE_MODEL") or "").strip()
    if not name or model is None:
        return None
    if hasattr(model, "model_copy"):
        try:
            return model.model_copy(update={"model": name, "temperature": 0})
        except Exception:
            logger.debug("structured escalate model_copy failed", exc_info=True)
    return None


def _try_outlines_local(model: Any, schema: type[SchemaT], messages: list[BaseMessage]) -> SchemaT | None:
    """本地可安装 outlines 时走 JSON CFG；云端 API 默认跳过。"""
    if (os.getenv("COMPANION_OUTLINES_JSON") or "").strip() not in ("1", "true", "yes"):
        return None
    try:
        import outlines  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("outlines 未安装，跳过 CFG 解码")
        return None
    try:
        generator = outlines.generate.json(model, schema.model_json_schema())
        prompt = "\n".join(str(getattr(m, "content", "") or "") for m in messages)
        raw = generator(prompt)
        return parse_json_with_schema(str(raw), schema)
    except Exception as exc:
        logger.warning("outlines JSON 生成失败: %s", exc)
        return None


def _run_attempts(
    model: Any,
    schema: type[SchemaT],
    msgs: list[BaseMessage],
    *,
    repairs: int,
    label: str,
) -> SchemaT:
    last_error: Exception | None = None
    for attempt in range(repairs + 1):
        last_error = None
        for method in structured_methods():
            try:
                return _invoke_once(model, schema, msgs, method)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "structured output method=%s failed label=%s attempt=%s: %s",
                    method,
                    label,
                    attempt,
                    exc,
                )
        if attempt >= repairs:
            break
        err_text = str(last_error or "格式不符合 Schema")
        if isinstance(last_error, ValidationError):
            err_text = json.dumps(last_error.errors(), ensure_ascii=False)
        msgs.append(
            HumanMessage(
                content=(
                    f"【格式校验失败】{err_text}\n"
                    "请严格按 system 末尾【结构化输出】中的 JSON Schema 重输出，只输出 JSON。"
                )
            )
        )
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"structured output 未返回结果 label={label}")


def invoke_structured_llm(
    model: Any,
    schema: type[SchemaT],
    messages: list[BaseMessage],
    *,
    max_repair: int | None = None,
    label: str = "",
    fallback: SchemaT | _FALLBACK | None = None,
) -> SchemaT:
    """优先 API json_schema / function_calling；失败回灌；仍失败可升模或 fallback。"""
    if model is None:
        if fallback is not None:
            return fallback() if callable(fallback) else fallback
        raise RuntimeError("structured output: model is None")
    repairs = structured_max_repair() if max_repair is None else max(0, min(int(max_repair), 5))

    cfg = _try_outlines_local(model, schema, messages)
    if cfg is not None:
        return cfg

    msgs = list(messages)
    try:
        return _run_attempts(model, schema, msgs, repairs=repairs, label=label)
    except Exception as primary_exc:
        escalated = _escalate_model(model)
        if escalated is not None:
            try:
                logger.info("structured output escalate model label=%s", label)
                return _run_attempts(escalated, schema, list(messages), repairs=1, label=f"{label}+escalate")
            except Exception:
                logger.warning("structured escalate failed label=%s", label, exc_info=True)
        if fallback is not None:
            logger.warning(
                "structured output fallback label=%s after %s",
                label,
                type(primary_exc).__name__,
            )
            return fallback() if callable(fallback) else fallback
        raise primary_exc
