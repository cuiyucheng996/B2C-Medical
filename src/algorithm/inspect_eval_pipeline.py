"""评测台：基于 Inspect AI 的 Agent 测评流水线（进程内 eval，日志落盘）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configs.data_paths import INSPECT_EVAL_LOG_DIR

__all__ = [
    "inspect_status",
    "list_tasks",
    "run_inspect_eval",
    "summarize_eval_log",
    "load_eval_report",
]


def _ensure_log_dir() -> Path:
    INSPECT_EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return INSPECT_EVAL_LOG_DIR


def list_tasks() -> list[dict[str, Any]]:
    from algorithm.inspect_companion_tasks import TASK_REGISTRY

    rows: list[dict[str, Any]] = []
    for key, meta in TASK_REGISTRY.items():
        rows.append(
            {
                "id": key,
                "title": meta.get("title") or key,
                "description": meta.get("description") or "",
                "default_limit": meta.get("default_limit"),
            }
        )
    return rows


def inspect_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "log_dir": str(_ensure_log_dir()),
        "tasks": list_tasks(),
        "inspect_ai": {"installed": False, "version": ""},
        "recent_logs": [],
        "ready": False,
        "blocking": [],
    }
    try:
        import inspect_ai

        out["inspect_ai"] = {
            "installed": True,
            "version": str(getattr(inspect_ai, "__version__", "") or ""),
        }
    except ImportError as exc:
        out["blocking"].append(f"inspect_ai 未安装：{exc}")
        return out

    try:
        from configs.data_paths import INTENT_CATALOG_PATH

        if not INTENT_CATALOG_PATH.is_file():
            out["blocking"].append(f"意图清单缺失：{INTENT_CATALOG_PATH}")
    except Exception as exc:
        out["blocking"].append(str(exc))

    try:
        from inspect_ai.log import list_eval_logs

        logs = list_eval_logs(log_dir=str(_ensure_log_dir()), recursive=True)
        recent: list[dict[str, Any]] = []
        for path in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
            recent.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                }
            )
        out["recent_logs"] = recent
    except Exception as exc:
        out["blocking"].append(f"列举日志失败：{exc}")

    out["ready"] = not out["blocking"]
    return out


def _metric_dict(score_row: Any) -> dict[str, float]:
    metrics = getattr(score_row, "metrics", None) or {}
    out: dict[str, float] = {}
    for key, val in metrics.items():
        try:
            out[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _location_path(location: Any) -> str:
    if location is None:
        return ""
    if isinstance(location, str):
        return location
    for attr in ("name", "path", "file"):
        val = getattr(location, attr, None)
        if isinstance(val, str) and val:
            return val
    return str(location)


def _sample_score_row(scores: Any) -> tuple[str | None, str]:
    if not scores:
        return None, ""
    if isinstance(scores, dict):
        for val in scores.values():
            if hasattr(val, "value"):
                return str(val.value), str(getattr(val, "explanation", "") or "")
        return None, ""
    nested = getattr(scores, "scores", None)
    if nested:
        first = nested[0]
        return str(getattr(first, "value", "") or ""), str(getattr(first, "explanation", "") or "")
    return None, ""


def summarize_eval_log_obj(el: Any) -> dict[str, Any]:
    results = getattr(el, "results", None)
    scores = list(results.scores or []) if results else []
    headline = _metric_dict(scores[0]) if scores else {}
    samples: list[dict[str, Any]] = []
    for sample in el.samples or []:
        score_val, explanation = _sample_score_row(sample.scores)
        completion = ""
        if sample.output is not None:
            completion = str(getattr(sample.output, "completion", "") or "")
        samples.append(
            {
                "id": sample.id,
                "input": sample.input,
                "target": sample.target,
                "output": completion[:500],
                "score": score_val,
                "explanation": explanation,
            }
        )
    ev = el.eval
    return {
        "log_path": _location_path(getattr(el, "location", "")),
        "task": getattr(ev, "task", ""),
        "model": getattr(ev, "model", ""),
        "created": getattr(ev, "created", ""),
        "metrics": headline,
        "samples": samples,
        "status": str(getattr(results, "status", "") or ""),
    }


def summarize_eval_log(log_path: str | Path) -> dict[str, Any]:
    from inspect_ai.log import read_eval_log

    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"日志不存在：{path}")
    el = read_eval_log(str(path))
    out = summarize_eval_log_obj(el)
    out["log_path"] = str(path)
    return out


def load_eval_report(name: str) -> dict[str, Any]:
    safe = (name or "").strip()
    if not safe.endswith(".eval"):
        safe = f"{safe}.eval"
    if ".." in safe or "/" in safe or "\\" in safe:
        raise ValueError("非法日志名")
    path = _ensure_log_dir() / safe
    if not path.is_file():
        for hit in _ensure_log_dir().rglob(safe):
            if hit.is_file():
                path = hit
                break
    return summarize_eval_log(path)


def run_inspect_eval(
    task_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    from inspect_ai import eval

    from algorithm.inspect_companion_tasks import TASK_REGISTRY

    meta = TASK_REGISTRY.get(task_id)
    if meta is None:
        raise ValueError(f"未知任务：{task_id}")
    log_dir = _ensure_log_dir()
    factory = meta["factory"]
    kwargs: dict[str, Any] = {
        "model": None,
        "display": "none",
        "log_dir": str(log_dir),
    }
    if limit is not None and limit > 0:
        kwargs["limit"] = int(limit)
    started = datetime.now(timezone.utc).isoformat()
    try:
        logs = eval(factory(), **kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "task": task_id,
            "started": started,
            "error": str(exc),
            "log_dir": str(log_dir),
        }
    if not logs:
        return {
            "ok": False,
            "task": task_id,
            "started": started,
            "error": "eval 未返回日志",
            "log_dir": str(log_dir),
        }
    el = logs[0]
    summary = summarize_eval_log_obj(el)
    log_file = _location_path(getattr(el, "location", ""))
    passed = sum(1 for s in summary.get("samples") or [] if s.get("score") == "C")
    total = len(summary.get("samples") or [])
    return {
        "ok": True,
        "task": task_id,
        "started": started,
        "log_dir": str(log_dir),
        "log_file": log_file,
        "passed": passed,
        "total": total,
        "metrics": summary.get("metrics") or {},
        "summary": summary,
    }
