"""中文文本纠错：主项目 AgentLoop harness + CorrectionResult 结构化输出。"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
_B2C_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()

import dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from prompt import load_prompt
from middleware import spell_check_middleware
from agent.confidence import (
    LOW_CONFIDENCE_THRESHOLD,
    calculate_confidence,
    get_rouge_scorer,
)
from agent.config import SpellCheckConfig
from agent.long_text import DEFAULT_MAX_CHUNK_LENGTH, merge_corrections, split_long_text
from schema.models import CorrectionResult
from tools.spell_check_tools import (
    bert_spell_correct,
    check_common_errors,
    query_dictionary,
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv(_B2C_ROOT / ".env")

SYSTEM_PROMPT = load_prompt("spell_check_system")

__all__ = [
    "SpellCheckAgent",
    "SpellCheckConfig",
    "correct_text",
    "get_spell_check_agent",
]


def _resolve_llm(config: SpellCheckConfig | None = None) -> Any:
    """优先 DeepSeek；若仅配置 OPENAI_* 则走 OpenAI 兼容网关。"""
    cfg = config or SpellCheckConfig()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        model = cfg.model or f"openai:{os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}"
        if not model.startswith("openai:"):
            model = f"openai:{model}"
        return init_chat_model(
            model,
            api_key=deepseek_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        model = cfg.model or "openai:gpt-4o"
        if not model.startswith("openai:"):
            model = f"openai:{model}"
        return init_chat_model(
            model,
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )

    raise RuntimeError(
        "未配置 LLM：请在项目根 .env 设置 DEEPSEEK_API_KEY"
    )


def _build_tools(config: SpellCheckConfig) -> list:
    if not config.use_tools:
        return []
    tools = [check_common_errors, query_dictionary]
    if config.use_bert_tool:
        tools.insert(0, bert_spell_correct)
    return tools


def _chat_model(config: SpellCheckConfig) -> Any:
    try:
        from src.llm.chat import get_chat_model

        model = get_chat_model()
        if model is not None:
            return model
    except Exception:
        logger.debug("uvgraph get_chat_model unavailable; use spell-check LLM", exc_info=True)
    return _resolve_llm(config)


def _invoke_spell_agent(config: SpellCheckConfig, chunk: str) -> CorrectionResult:
    """create_agent + 中间件调度工具，结束后 structured output。"""
    tools = _build_tools(config)
    agent = create_agent(
        model=_chat_model(config),
        tools=tools,
        middleware=spell_check_middleware(),
        response_format=ToolStrategy(CorrectionResult),
        name="spell_check",
    )
    result = agent.invoke({"messages": [HumanMessage(content=f"请纠正以下中文文本：{chunk}")]})
    if not isinstance(result, dict):
        raise RuntimeError("spell check agent 无返回")
    structured = result.get("structured_response")
    if isinstance(structured, CorrectionResult):
        return structured
    if isinstance(structured, dict):
        return CorrectionResult.model_validate(structured)
    raise RuntimeError("spell check agent 无结构化结果")


def create_spell_check_agent(config: SpellCheckConfig | None = None) -> SpellCheckAgent:
    return SpellCheckAgent(config=config)


def get_spell_check_agent() -> SpellCheckAgent:
    return _get_default_agent()


def _apply_rouge_confidence(
    result: CorrectionResult,
    original: str,
    *,
    scorer=None,
) -> CorrectionResult:
    rouge_confidence = calculate_confidence(original, result.corrected_text, scorer=scorer)
    result.confidence = min(result.confidence, rouge_confidence)
    return result


def _correct_chunk(config: SpellCheckConfig, chunk: str, *, scorer=None) -> CorrectionResult:
    try:
        result = _invoke_spell_agent(config, chunk)
        return _apply_rouge_confidence(result, chunk, scorer=scorer)
    except Exception as exc:
        logger.exception("spell_check chunk failed")
        return CorrectionResult(
            original_text=chunk,
            corrected_text=chunk,
            errors=[],
            confidence=0.0,
            explanation=f"纠错失败: {exc}",
        )


def _reject_low_confidence(text: str, result: CorrectionResult) -> CorrectionResult:
    return CorrectionResult(
        original_text=text,
        corrected_text=text,
        errors=[],
        confidence=result.confidence,
        explanation=(
            f"置信度不足({result.confidence:.2f})，保留原文。建议人工审核。"
        ),
        chunks=result.chunks,
    )


class SpellCheckAgent:
    """纠错外壳：分块 + create_agent 中间件 + 置信度门。"""

    def __init__(
        self,
        config: SpellCheckConfig | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        use_tools: bool = True,
        use_bert_tool: bool = True,
        max_chunk_length: int = DEFAULT_MAX_CHUNK_LENGTH,
        require_high_confidence: bool = True,
    ) -> None:
        if config is None:
            config = SpellCheckConfig(
                model=model,
                temperature=temperature,
                use_tools=use_tools,
                use_bert_tool=use_bert_tool,
                max_chunk_length=max_chunk_length,
                require_high_confidence=require_high_confidence,
            )
        self.config = config
        self.rouge_scorer = get_rouge_scorer()

    def split_long_text(self, text: str, max_length: int | None = None) -> list[str]:
        return split_long_text(text, max_length=max_length or self.config.max_chunk_length)

    def calculate_confidence(self, original: str, corrected: str) -> float:
        return calculate_confidence(original, corrected, scorer=self.rouge_scorer)

    def merge_corrections(
        self,
        chunks: list[str],
        results: list[CorrectionResult],
    ) -> CorrectionResult:
        return merge_corrections(chunks, results)

    def correct(
        self,
        text: str,
        *,
        require_high_confidence: bool | None = None,
        max_chunk_length: int | None = None,
    ) -> CorrectionResult:
        return _run_correction(
            text,
            config=self.config,
            scorer=self.rouge_scorer,
            max_chunk_length=max_chunk_length or self.config.max_chunk_length,
            require_high_confidence=(
                self.config.require_high_confidence
                if require_high_confidence is None
                else require_high_confidence
            ),
        )


def _run_correction(
    text: str,
    *,
    config: SpellCheckConfig,
    scorer,
    max_chunk_length: int,
    require_high_confidence: bool,
) -> CorrectionResult:
    text = text.strip()
    if not text:
        return CorrectionResult(
            original_text="",
            corrected_text="",
            errors=[],
            confidence=1.0,
            explanation="空文本无需纠错",
        )

    chunks = split_long_text(text, max_length=max_chunk_length)
    results = [_correct_chunk(config, chunk, scorer=scorer) for chunk in chunks]

    if len(chunks) > 1:
        final_result = merge_corrections(chunks, results)
    else:
        final_result = results[0]
        final_result.original_text = text

    if require_high_confidence and final_result.confidence < LOW_CONFIDENCE_THRESHOLD:
        return _reject_low_confidence(text, final_result)

    return final_result


_default_agent: SpellCheckAgent | None = None


def _get_default_agent() -> SpellCheckAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = SpellCheckAgent()
    return _default_agent


def correct_text(
    text: str,
    *,
    max_chunk_length: int = DEFAULT_MAX_CHUNK_LENGTH,
    require_high_confidence: bool = True,
    agent: SpellCheckAgent | None = None,
) -> CorrectionResult:
    """函数式入口：默认复用进程内单例 SpellCheckAgent。"""
    runner = agent or _get_default_agent()
    return runner.correct(
        text,
        require_high_confidence=require_high_confidence,
        max_chunk_length=max_chunk_length,
    )


def _print_result(case_id: str, name: str, text: str, result: CorrectionResult) -> None:
    print(f"\n【{case_id}】{name}")
    print(f"原文: {text}")
    print(f"纠正: {result.corrected_text}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"说明: {result.explanation}")
    if result.chunks:
        print(f"分块数: {len(result.chunks)}")
    if result.errors:
        print("检测到的错误:")
        for idx, error in enumerate(result.errors, 1):
            print(
                f"  {idx}. [{error.error_type}] 位置{error.position}: "
                f"'{error.error_span}' → '{error.correction}' ({error.reason})"
            )
    print("-" * 80)


if __name__ == "__main__":
    from agent.fixtures import SPELL_CHECK_SAMPLES

    corrector = SpellCheckAgent(temperature=0.2, use_tools=True)
    print(
        "ROUGE 置信度示例:",
        f"{corrector.calculate_confidence('我喜你', '我喜欢你'):.4f}",
    )

    for sample in SPELL_CHECK_SAMPLES:
        out = corrector.correct(sample.text)
        _print_result(sample.id, sample.name, sample.text, out)
