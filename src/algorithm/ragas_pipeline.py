"""RAGAS 离线评测：sources 入库 → qa 检索（+可选生成）→ ragas 指标。"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from configs.paths_b2c import B2C_ROOT, ensure_b2c_runtime_path

logger = logging.getLogger(__name__)

__all__ = [
    "REPORTS_DIR",
    "default_paths",
    "ingest_rows",
    "load_jsonl",
    "ragas_status",
    "run_ragas_eval",
    "save_jsonl_text",
]

EVAL_ROOT = B2C_ROOT / "tests" / "eval_ragas"
REPORTS_DIR = EVAL_ROOT / "reports"


def default_paths() -> dict[str, str]:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sources = EVAL_ROOT / "sources.jsonl"
    qa = EVAL_ROOT / "qa.jsonl"
    if not sources.is_file():
        sources.write_text(
            json.dumps(
                {
                    "title": "高血压生活建议",
                    "text": "高血压患者应低盐饮食，每日食盐建议不超过5克，并规律监测血压。",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    if not qa.is_file():
        qa.write_text(
            json.dumps(
                {
                    "question": "高血压患者为什么要控盐？",
                    "ground_truth": "应低盐，每日食盐不超过约5克，并监测血压。",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"sources": str(sources), "qa": str(qa), "reports": str(REPORTS_DIR)}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def save_jsonl_text(path: str | Path, text: str) -> int:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = (text or "").replace("\r\n", "\n").strip()
    file_path.write_text(normalized + ("\n" if normalized else ""), encoding="utf-8")
    return len(load_jsonl(file_path))


def _list_reports() -> list[dict[str, Any]]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS_DIR.glob("ragas_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "mtime": p.stat().st_mtime} for p in files[:30]]


def ragas_status() -> dict[str, Any]:
    from algorithm.knowledge_pipe import pipe_status

    paths = default_paths()
    sources_n = len(load_jsonl(paths["sources"]))
    qa_n = len(load_jsonl(paths["qa"]))
    pipe = pipe_status()
    return {
        "paths": paths,
        "sources_lines": sources_n,
        "qa_lines": qa_n,
        "reports": _list_reports(),
        "knowledge_pipe": pipe,
    }


def ingest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from algorithm.knowledge_pipe import ingest

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        title = str(row.get("title") or f"source-{idx + 1}").strip()
        text = str(row.get("text") or "").strip()
        query = str(row.get("query") or "").strip()
        url = str(row.get("url") or "").strip()
        if not any([text, query, url]):
            results.append({"ok": False, "title": title, "error": "需要 text / query / url 至少一项"})
            continue
        try:
            payload = ingest(text=text, title=title, query=query, url=url)
            results.append({"ok": True, "title": title, **payload})
        except Exception as exc:
            logger.warning("ragas ingest row failed: %s", exc)
            results.append({"ok": False, "title": title, "error": str(exc)})
    return results


def _ensure_ragas_import() -> None:
    key = "langchain_community.chat_models.vertexai"
    if key in sys.modules:
        return
    stub = ModuleType(key)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[key] = stub


def _retrieve_hits(question: str, channel: str, top_k: int) -> list[dict[str, Any]]:
    from algorithm.pdf_kb import retrieve_local_hits
    from algorithm.ragflow_kb import ping_ragflow, retrieve_hits

    q = (question or "").strip()
    if not q:
        return []
    ch = (channel or "local").strip().lower()
    if ch == "ragflow":
        if not ping_ragflow().get("ok"):
            raise RuntimeError("RAGFlow 不可用，请检查配置或服务")
        from algorithm.ragflow_kb import _retrieve_remote_hits

        hits = _retrieve_remote_hits(q, top_k=top_k) or retrieve_hits(q, top_k=top_k)
        for row in hits:
            row.setdefault("channel", "ragflow")
        return hits
    return retrieve_local_hits(q, top_k=top_k)


def _generate_answer(question: str, contexts: list[str]) -> str:
    ensure_b2c_runtime_path()
    from langchain_core.messages import HumanMessage, SystemMessage

    from schema.llm import get_deepseek_llm

    llm = get_deepseek_llm()
    ctx_block = "\n".join(f"- {c}" for c in contexts if c)[:6000]
    resp = llm.invoke(
        [
            SystemMessage(content="你是医疗科普助手。仅根据给定资料简要回答；资料不足请说明不确定。"),
            HumanMessage(content=f"资料：\n{ctx_block or '（无）'}\n\n问题：{question}"),
        ]
    )
    return str(getattr(resp, "content", "") or "").strip()


def _qa_fields(row: dict[str, Any]) -> tuple[str, str, str]:
    question = str(
        row.get("question") or row.get("user_input") or row.get("query") or ""
    ).strip()
    reference = str(
        row.get("ground_truth") or row.get("reference") or row.get("answer") or ""
    ).strip()
    preset = str(row.get("response") or "").strip()
    return question, reference, preset


def _build_ragas_llm_embeddings():
    _ensure_ragas_import()
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from schema.llm import get_deepseek_llm

    llm = LangchainLLMWrapper(get_deepseek_llm())
    model_name = "BAAI/bge-small-zh-v1.5"
    try:
        from src.configs import get_yaml_config

        block = get_yaml_config().get("embedding") if isinstance(get_yaml_config().get("embedding"), dict) else {}
        model_name = str(block.get("model") or model_name)
    except Exception:
        pass
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    )
    return llm, embeddings


def _run_ragas_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    _ensure_ragas_import()
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.collections import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    if not samples:
        raise ValueError("无有效评测样本")
    llm, embeddings = _build_ragas_llm_embeddings()
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=str(s["user_input"]),
                retrieved_contexts=list(s["retrieved_contexts"]),
                response=str(s["response"]),
                reference=str(s.get("reference") or ""),
            )
            for s in samples
        ]
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    aggregate: dict[str, float] = {}
    per_row: list[dict[str, Any]] = []
    for col in frame.columns:
        series = frame[col]
        if series.dtype.kind in "fiu":
            aggregate[col] = float(series.mean())
    for _, row in frame.iterrows():
        per_row.append({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.to_dict().items()})
    return {"aggregate": aggregate, "per_sample": per_row}


def run_ragas_eval(
    *,
    qa_path: str,
    channel: str = "local",
    top_k: int = 5,
    generate_answer: bool = True,
    save_report: bool = True,
) -> dict[str, Any]:
    rows = load_jsonl(qa_path)
    if not rows:
        raise ValueError("qa.jsonl 为空")

    trace: list[dict[str, Any]] = []
    ragas_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for row in rows:
        question, reference, preset = _qa_fields(row)
        if not question:
            trace.append({"question": "", "error": "缺少 question", "hits": []})
            continue
        t_ret = time.perf_counter()
        hits = _retrieve_hits(question, channel, max(1, min(20, int(top_k))))
        retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)
        contexts = [str(h.get("text") or h.get("content") or "") for h in hits if h]
        response = preset
        generate_ms = 0.0
        if generate_answer:
            t_gen = time.perf_counter()
            try:
                response = _generate_answer(question, contexts)
            except Exception as exc:
                response = ""
                trace.append(
                    {
                        "question": question,
                        "error": f"生成失败: {exc}",
                        "hits": hits,
                        "retrieval_ms": retrieval_ms,
                        "generate_ms": 0,
                    }
                )
                continue
            generate_ms = round((time.perf_counter() - t_gen) * 1000, 1)
        elif not response:
            response = reference

        trace.append(
            {
                "question": question,
                "reference": reference,
                "response": response,
                "hits": hits,
                "retrieval_ms": retrieval_ms,
                "generate_ms": generate_ms,
            }
        )
        if not response:
            continue
        ragas_rows.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts or ["（无检索结果）"],
                "response": response,
                "reference": reference,
            }
        )

    out: dict[str, Any] = {
        "channel": channel,
        "top_k": top_k,
        "generate_answer": generate_answer,
        "trace": trace,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }

    if not ragas_rows:
        out["scores"] = {}
        out["ragas_error"] = "没有可评分样本（检查 question / 生成回答 / API Key）"
        return out

    try:
        scored = _run_ragas_metrics(ragas_rows)
        out["scores"] = scored.get("aggregate") or {}
        out["per_sample_scores"] = scored.get("per_sample") or []
    except Exception as exc:
        logger.exception("ragas evaluate failed")
        out["scores"] = {}
        out["ragas_error"] = str(exc)

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"ragas_{channel}_{stamp}.json"
        path = REPORTS_DIR / name
        payload = {**out, "report": name}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        out["report"] = name
        out["report_path"] = str(path)

    return out
