"""8.2 注册项：ToolEntry（ToolSpec + executor / subagent + async execute）。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from ...schemas.agent_protocol import EmitFn, SubAgent, SubAgentResult, SubAgentTask
from .tool_spec import ToolSpec, spec_to_langchain_tool

ToolExecutor = Callable[[dict[str, Any]], Any]

__all__ = [
    "ToolEntry",
    "ToolExecutor",
    "ToolMeta",
]


@dataclass(slots=True)
class ToolMeta:
    """后台展示与问诊阶段元数据（不进 OpenAI schema）。"""

    title: str
    summary: str
    phases: tuple[int, ...] = ()
    category: str = "lookup"
    operation_hint: str = ""


class ToolEntry:
    """工具注册项。"""

    def __init__(
        self,
        spec: ToolSpec,
        executor: ToolExecutor | None = None,
        subagent: SubAgent | None = None,
        *,
        meta: ToolMeta | None = None,
        langchain_tool: BaseTool | None = None,
    ) -> None:
        self.spec = spec
        self.executor = executor
        self.subagent = subagent
        # 项目扩展：后台元数据 + 预构建 LangChain Tool
        self.meta = meta
        self.langchain_tool = langchain_tool

    @property
    def name(self) -> str:
        return self.spec.name

    async def execute(self, args: dict[str, Any], emit: EmitFn | None = None) -> SubAgentResult:
        """执行工具：SubAgent 走 Task/Result；executor 走普通函数。"""
        if self.subagent is not None:
            task = SubAgentTask(
                instruction=str(args.get("instruction") or ""),
                refs=list(args.get("refs") or []),
                resume_token=args.get("resume_token"),
                context=dict(args),
            )
            result = self.subagent.run(task, emit)
            if inspect.isawaitable(result):
                return await result
            return result

        if self.executor is not None:
            try:
                result = self.executor(dict(args))
                if inspect.isawaitable(result):
                    result = await result
            except Exception as exc:
                return SubAgentResult.failed(f"工具 {self.name} 执行失败：{exc}")
            return self._normalize_executor_result(result)

        return SubAgentResult.failed(f"工具 {self.name} 未配置执行器")

    def execute_sync(self, args: dict[str, Any], emit: EmitFn | None = None) -> SubAgentResult:
        """同步入口：LangChain invoke / 无事件循环场景。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(args, emit))
        return self._execute_sync_blocking(args, emit)

    def _execute_sync_blocking(self, args: dict[str, Any], emit: EmitFn | None) -> SubAgentResult:
        if self.subagent is not None:
            task = SubAgentTask(
                instruction=str(args.get("instruction") or ""),
                refs=list(args.get("refs") or []),
                resume_token=args.get("resume_token"),
                context=dict(args),
            )
            return self.subagent.run(task, emit)

        if self.executor is not None:
            try:
                return self._normalize_executor_result(self.executor(dict(args)))
            except Exception as exc:
                return SubAgentResult.failed(f"工具 {self.name} 执行失败：{exc}")

        return SubAgentResult.failed(f"工具 {self.name} 未配置执行器")

    @staticmethod
    def _normalize_executor_result(result: Any) -> SubAgentResult:
        if isinstance(result, SubAgentResult):
            return result
        if isinstance(result, dict):
            output = str(result.get("output", result))
            meta = {k: v for k, v in result.items() if k != "output"}
            if meta:
                return SubAgentResult(status="done", output=output, snapshot=meta)
            return SubAgentResult.done(output)
        if isinstance(result, (list, tuple)):
            return SubAgentResult.done(json.dumps(result, ensure_ascii=False))
        return SubAgentResult.done(str(result))

    def to_langchain_tool(self) -> BaseTool:
        if self.langchain_tool is not None:
            return self.langchain_tool
        return spec_to_langchain_tool(self.spec, fallback=None, invoke=self._langchain_invoke)

    def _langchain_invoke(self, **kwargs: Any) -> Any:
        if self.langchain_tool is not None:
            return self.langchain_tool.invoke(kwargs)
        result = self.execute_sync(kwargs, emit=None)
        if result.status == "failed":
            return result.error or result.output
        return result.output

    def to_summary_dict(self, *, category_labels: dict[str, str] | None = None) -> dict[str, Any]:
        labels = category_labels or {}
        meta = self.meta
        category = meta.category if meta else "lookup"
        return {
            "name": self.spec.name,
            "title": meta.title if meta else self.spec.name,
            "summary": meta.summary if meta else "",
            "phases": list(meta.phases) if meta else [],
            "category": category,
            "category_label": labels.get(category, category),
            "operation_hint": meta.operation_hint if meta else "",
            "needs_confirmation": self.spec.needs_confirmation,
            "visible_to": self.spec.visible_to,
            "kind": "subagent" if self.subagent is not None else "executor",
        }

    def to_detail_dict(self, *, category_labels: dict[str, str] | None = None) -> dict[str, Any]:
        detail = self.spec.to_detail_dict()
        detail.update(self.to_summary_dict(category_labels=category_labels))
        detail["llm_description"] = self.spec.description.strip()
        return detail
