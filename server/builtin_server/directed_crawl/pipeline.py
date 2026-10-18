"""定向采集流水线：步骤可开关、参数可覆盖。"""

from __future__ import annotations

import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from server.builtin_server.directed_crawl import config as dc_config
from server.builtin_server.directed_crawl.fetch_page import extract_text_from_html, fetch_html_raw

_BUILTIN_PAGE_FETCH_STEPS: list[dict[str, Any]] = [
    {"id": "whitelist", "title": "白名单 host 校验", "enabled": True, "params": {}},
    {
        "id": "http_get",
        "title": "HTTP 拉取 HTML",
        "enabled": True,
        "params": {
            "max_bytes": 400_000,
            "timeout_sec": 25,
            "user_agent": "b2c-directed-crawl/1.0",
        },
    },
    {
        "id": "extract_text",
        "title": "去标签抽正文",
        "enabled": True,
        "params": {"max_output_chars": 12_000},
    },
    {"id": "persist_page", "title": "写入 crawl_data/pages", "enabled": True, "params": {}},
    {"id": "save_job", "title": "写入 jobs JSON", "enabled": True, "params": {"text_preview_chars": 2000}},
]


def pipeline_catalog() -> dict[str, Any]:
    sources = dc_config.load_sources()
    raw = sources.get("pipelines") if isinstance(sources.get("pipelines"), dict) else {}
    page = raw.get("page_fetch") if isinstance(raw.get("page_fetch"), dict) else {}
    steps = page.get("steps") if isinstance(page.get("steps"), list) else None
    return {
        "page_fetch": {
            "title": str(page.get("title") or "白名单网页抓取"),
            "steps": _normalize_steps(steps or _BUILTIN_PAGE_FETCH_STEPS),
        }
    }


def _normalize_steps(steps: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        if not sid:
            continue
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        out.append(
            {
                "id": sid,
                "title": str(item.get("title") or sid),
                "enabled": bool(item.get("enabled", True)),
                "params": dict(params),
            }
        )
    return out or deepcopy(_BUILTIN_PAGE_FETCH_STEPS)


def merge_step_overrides(
    defaults: list[dict[str, Any]], overrides: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if not overrides:
        return deepcopy(defaults)
    by_id = {s["id"]: deepcopy(s) for s in defaults}
    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        sid = str(ov.get("id") or "").strip()
        if not sid or sid not in by_id:
            continue
        if "enabled" in ov:
            by_id[sid]["enabled"] = bool(ov["enabled"])
        op = ov.get("params")
        if isinstance(op, dict):
            by_id[sid]["params"] = {**by_id[sid].get("params", {}), **op}
    return list(by_id.values())


def _step_enabled(steps: list[dict[str, Any]], step_id: str) -> bool:
    for s in steps:
        if s.get("id") == step_id:
            return bool(s.get("enabled", True))
    return False


def _step_params(steps: list[dict[str, Any]], step_id: str) -> dict[str, Any]:
    for s in steps:
        if s.get("id") == step_id:
            p = s.get("params")
            return dict(p) if isinstance(p, dict) else {}
    return {}


def run_page_fetch_pipeline(
    *,
    url: str,
    profile: str = "health_guide",
    session_id: str = "",
    steps_override: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
    save_job_fn: Any,
) -> dict[str, Any]:
    """执行 page_fetch 流水线，返回 trace + 与 /fetch 兼容的字段。"""
    t0 = time.perf_counter()
    target = (url or "").strip()
    if not target:
        return {"ok": False, "error": "缺少 url"}

    defaults = pipeline_catalog()["page_fetch"]["steps"]
    steps = merge_step_overrides(defaults, steps_override)
    trace: list[dict[str, Any]] = []
    ctx: dict[str, Any] = {"url": target, "profile": profile, "session_id": session_id}

    def _trace(step_id: str, title: str, **extra: Any) -> None:
        trace.append({"id": step_id, "title": title, **extra})

    sources = dc_config.load_sources()
    prof = (sources.get("profiles") or {}).get(profile) or {}
    disclaimer = str(prof.get("disclaimer") or "公开信息仅供参考。")
    ctx["disclaimer"] = disclaimer

    # whitelist
    if _step_enabled(steps, "whitelist"):
        t1 = time.perf_counter()
        if not dc_config.host_allowed(target):
            _trace(
                "whitelist",
                "白名单 host 校验",
                ok=False,
                skipped=False,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
                error="url 不在白名单 host 内",
            )
            return {"ok": False, "error": "url 不在白名单 host 内", "url": target, "trace": trace}
        _trace(
            "whitelist",
            "白名单 host 校验",
            ok=True,
            skipped=False,
            elapsed_ms=int((time.perf_counter() - t1) * 1000),
        )
    else:
        _trace("whitelist", "白名单 host 校验", ok=True, skipped=True, elapsed_ms=0)

    # http_get
    raw: bytes | None = None
    charset = "utf-8"
    if _step_enabled(steps, "http_get"):
        t1 = time.perf_counter()
        hp = _step_params(steps, "http_get")
        max_bytes = int(hp.get("max_bytes") or dc_config.max_fetch_bytes())
        timeout_sec = float(hp.get("timeout_sec") or 25)
        user_agent = str(hp.get("user_agent") or "b2c-directed-crawl/1.0")
        try:
            raw, charset = fetch_html_raw(
                target,
                max_bytes=max_bytes,
                timeout_sec=timeout_sec,
                user_agent=user_agent,
            )
            ctx["html_bytes"] = len(raw)
            ctx["charset"] = charset
            _trace(
                "http_get",
                "HTTP 拉取 HTML",
                ok=True,
                skipped=False,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
                html_bytes=len(raw),
            )
        except Exception as exc:
            _trace(
                "http_get",
                "HTTP 拉取 HTML",
                ok=False,
                skipped=False,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
                error=str(exc),
            )
            return {"ok": False, "error": str(exc), "url": target, "profile": profile, "trace": trace}
    else:
        _trace("http_get", "HTTP 拉取 HTML", ok=True, skipped=True, elapsed_ms=0)

    # extract_text
    text = ""
    if _step_enabled(steps, "extract_text"):
        if raw is None:
            return {
                "ok": False,
                "error": "未执行 http_get，无法抽正文",
                "trace": trace,
            }
        t1 = time.perf_counter()
        ep = _step_params(steps, "extract_text")
        hp = _step_params(steps, "http_get")
        max_bytes = int(hp.get("max_bytes") or dc_config.max_fetch_bytes())
        max_out = int(ep.get("max_output_chars") or 12_000)
        try:
            html = raw.decode(charset, errors="replace")
            text = extract_text_from_html(
                html, max_bytes_hint=max_bytes, max_output_chars=max_out
            )
            ctx["text"] = text
            _trace(
                "extract_text",
                "去标签抽正文",
                ok=True,
                skipped=False,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
                text_chars=len(text),
            )
        except Exception as exc:
            _trace(
                "extract_text",
                "去标签抽正文",
                ok=False,
                skipped=False,
                elapsed_ms=int((time.perf_counter() - t1) * 1000),
                error=str(exc),
            )
            return {"ok": False, "error": str(exc), "url": target, "profile": profile, "trace": trace}
    else:
        _trace("extract_text", "去标签抽正文", ok=True, skipped=True, elapsed_ms=0)

    job_id = uuid.uuid4().hex[:16]
    text_path = ""
    if dry_run:
        _trace("persist_page", "写入 crawl_data/pages", ok=True, skipped=True, elapsed_ms=0, dry_run=True)
        _trace("save_job", "写入 jobs JSON", ok=True, skipped=True, elapsed_ms=0, dry_run=True)
        return {
            "ok": True,
            "dry_run": True,
            "profile": profile,
            "url": target,
            "text": text,
            "text_chars": len(text),
            "disclaimer": disclaimer,
            "trace": trace,
            "steps": steps,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    if _step_enabled(steps, "persist_page") and text:
        t1 = time.perf_counter()
        pages_dir = dc_config.cache_root() / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        text_path = str(pages_dir / f"{job_id}.txt")
        Path(text_path).write_text(f"url: {target}\n\n{text}", encoding="utf-8")
        _trace(
            "persist_page",
            "写入 crawl_data/pages",
            ok=True,
            skipped=False,
            elapsed_ms=int((time.perf_counter() - t1) * 1000),
            text_file=text_path,
        )
    else:
        _trace("persist_page", "写入 crawl_data/pages", ok=True, skipped=not _step_enabled(steps, "persist_page"), elapsed_ms=0)

    if _step_enabled(steps, "save_job"):
        t1 = time.perf_counter()
        sp = _step_params(steps, "save_job")
        preview_n = int(sp.get("text_preview_chars") or 2000)
        payload = {
            "job_id": job_id,
            "kind": "page_fetch",
            "profile": profile,
            "url": target,
            "session_id": session_id,
            "text_chars": len(text),
            "text_file": text_path or None,
            "text_preview": text[:preview_n] if text else "",
            "disclaimer": disclaimer,
            "pipeline_trace": trace,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        job_id = save_job_fn(payload)
        _trace(
            "save_job",
            "写入 jobs JSON",
            ok=True,
            skipped=False,
            elapsed_ms=int((time.perf_counter() - t1) * 1000),
            job_id=job_id,
        )
    else:
        _trace("save_job", "写入 jobs JSON", ok=True, skipped=True, elapsed_ms=0)

    out: dict[str, Any] = {
        "ok": True,
        "job_id": job_id,
        "profile": profile,
        "url": target,
        "text": text,
        "disclaimer": disclaimer,
        "trace": trace,
        "steps": steps,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    if text_path:
        out["text_file"] = text_path
    return out
