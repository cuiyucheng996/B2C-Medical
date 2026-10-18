"""定向采集：白名单抓页、影刀 Excel 入库、任务落盘。"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server.builtin_server.directed_crawl import config as dc_config
from server.builtin_server.directed_crawl.excel_ingest import ingest_excel_auto
from server.builtin_server.directed_crawl.pipeline import pipeline_catalog, run_page_fetch_pipeline
from server.builtin_server.directed_crawl.yingdao_runner import start_rpa_flow

logger = logging.getLogger(__name__)

_LATEST_POINTER = "latest_job_id.txt"


def health_payload() -> dict[str, str]:
    return {"status": "healthy", "service": "directed_crawl"}


def catalog() -> dict[str, Any]:
    sources = dc_config.load_sources()
    profiles = sources.get("profiles") or {}
    return {
        "ok": True,
        "sources_file": str(dc_config.sources_path()),
        "whitelist_hosts": sources.get("whitelist_hosts") or [],
        "rpa_output_dirs": sources.get("rpa_output_dirs") or [],
        "profiles": profiles,
        "seed_urls": sources.get("seed_urls") or [],
        "pipelines": pipeline_catalog(),
        "rpa_flows": sources.get("rpa_flows") or {},
    }


def _job_path(job_id: str) -> Path:
    return dc_config.cache_root() / "jobs" / f"{job_id}.json"


def _save_job(payload: dict[str, Any]) -> str:
    job_id = str(payload.get("job_id") or uuid.uuid4().hex[:16])
    payload["job_id"] = job_id
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path = _job_path(job_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dc_config.cache_root() / _LATEST_POINTER).write_text(job_id, encoding="utf-8")
    return job_id


def get_job(job_id: str) -> dict[str, Any]:
    path = _job_path(job_id)
    if not path.is_file():
        return {"ok": False, "error": "job 不存在", "job_id": job_id}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"ok": True, **data}


def get_latest_job() -> dict[str, Any]:
    ptr = dc_config.cache_root() / _LATEST_POINTER
    if not ptr.is_file():
        return {"ok": False, "error": "尚无任务"}
    job_id = ptr.read_text(encoding="utf-8").strip()
    return get_job(job_id)


def run_pipeline(
    *,
    url: str,
    profile: str = "health_guide",
    session_id: str = "",
    pipeline: str = "page_fetch",
    steps: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if pipeline != "page_fetch":
        return {"ok": False, "error": f"未知 pipeline: {pipeline}"}
    result = run_page_fetch_pipeline(
        url=url,
        profile=profile,
        session_id=session_id,
        steps_override=steps,
        dry_run=dry_run,
        save_job_fn=_save_job,
    )
    if not result.get("ok"):
        logger.warning("run_pipeline failed url=%s err=%s", url, result.get("error"))
    return result


def fetch_directed(*, url: str, profile: str = "health_guide", session_id: str = "") -> dict[str, Any]:
    return run_pipeline(url=url, profile=profile, session_id=session_id, dry_run=False)


def start_directed_rpa(
    *,
    profile: str,
    params: dict[str, Any] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    return start_rpa_flow(
        profile=profile,
        params=params,
        session_id=session_id,
        save_job_fn=_save_job,
    )


def _resolve_excel(path_str: str) -> Path | None:
    raw = (path_str or "").strip().strip('"')
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (dc_config.B2C_ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        return None
    if not dc_config.excel_path_allowed(p):
        return None
    return p


def start_crawl_webhook(body: dict[str, Any]) -> dict[str, Any]:
    """兼容影刀教程 POST /webhook/start-crawl（可指向本服务 8014 替代 n8n）。"""
    t0 = time.perf_counter()
    source = str(body.get("source") or body.get("来源") or "rpa").strip()
    status = str(body.get("status") or "finished").strip()
    job_type = str(body.get("job_type") or body.get("type") or "").strip()
    crawl_date = str(body.get("crawl_date") or body.get("采集日期") or "").strip()
    excel_path = str(body.get("excel_path") or body.get("final_excel") or "").strip()

    payload: dict[str, Any] = {
        "kind": "rpa_webhook",
        "source": source,
        "status": status,
        "job_type": job_type,
        "crawl_date": crawl_date,
        "raw": {k: v for k, v in body.items() if k not in {"excel_bytes"}},
    }
    excel_file = _resolve_excel(excel_path)
    if excel_file:
        try:
            archive_dir = dc_config.cache_root() / "excel"
            archived = archive_dir / f"{int(time.time())}_{excel_file.name}"
            shutil.copy2(excel_file, archived)
            payload["excel_archived"] = str(archived)
            payload["excel"] = ingest_excel_auto(excel_file, job_type=job_type)
            payload["excel_path"] = str(excel_file)
        except Exception as exc:
            logger.exception("excel ingest failed path=%s", excel_file)
            payload["excel_error"] = str(exc)
    elif excel_path:
        payload["excel_skipped"] = "路径无效或不在 rpa_output_dirs 白名单"
        payload["excel_path_requested"] = excel_path

    payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
    job_id = _save_job(payload)
    card = compose_companion_card(payload)
    return {"ok": True, "job_id": job_id, "card": card, **{k: v for k, v in payload.items() if k != "raw"}}


def ingest_excel_bytes(
    *,
    filename: str,
    data: bytes,
    job_type: str = "",
    source: str = "upload",
) -> dict[str, Any]:
    if not data:
        return {"ok": False, "error": "空文件"}
    inbox = dc_config.cache_root() / "inbox"
    safe = Path(filename or "upload.xlsx").name
    if not safe.lower().endswith((".xlsx", ".xlsm")):
        return {"ok": False, "error": "仅支持 .xlsx"}
    dest = inbox / f"{int(time.time())}_{safe}"
    dest.write_bytes(data)
    try:
        parsed = ingest_excel_auto(dest, job_type=job_type)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "saved": str(dest)}
    job_id = _save_job(
        {
            "kind": "excel_upload",
            "source": source,
            "excel_path": str(dest),
            "excel": parsed,
        }
    )
    return {"ok": True, "job_id": job_id, "excel_path": str(dest), "excel": parsed, "card": compose_companion_card({"excel": parsed})}


def compose_companion_card(job: dict[str, Any]) -> dict[str, Any]:
    """给陪聊 Agent 用的简短结构化摘要。"""
    excel = job.get("excel") if isinstance(job.get("excel"), dict) else None
    if excel and excel.get("kind") == "amazon_bsr":
        n = excel.get("main_count", 0)
        m = excel.get("id_count", 0)
        return {
            "title": "榜单采集结果",
            "summary": f"主数据 {n} 行，ASIN/RANK {m} 条（见 doc/External 榜单教程）。",
            "bullets": [
                f"主表预览 {len(excel.get('main_preview') or [])} 条已载入任务",
                "可结合用户体检/健康素养做选品解读，勿当作医疗建议",
            ],
            "disclaimer": "电商公开信息仅供参考，不构成医疗或投资建议。",
            "data_ref": "directed_crawl:latest",
        }
    if excel and excel.get("kind") == "amazon_product":
        return {
            "title": "单品深度采集",
            "summary": f"商品 {excel.get('goods_count', 0)} 条，评论 {excel.get('remarks_count', 0)} 条。",
            "bullets": ["详情与评论可用于痛点/卖点归纳", "回复用户时需标注来源与采集日期"],
            "disclaimer": "用户评价不代表医学结论。",
            "data_ref": "directed_crawl:latest",
        }
    if job.get("kind") == "page_fetch" or job.get("text_preview"):
        return {
            "title": "定向网页摘要",
            "summary": (job.get("text_preview") or "")[:280],
            "bullets": [f"来源：{job.get('url', '')}"],
            "disclaimer": job.get("disclaimer") or "公开资料仅供参考。",
            "data_ref": "directed_crawl:latest",
        }
    return {
        "title": "采集任务",
        "summary": str(job.get("status") or job.get("kind") or "已记录"),
        "bullets": [],
        "disclaimer": "公开信息仅供参考。",
        "data_ref": "directed_crawl:latest",
    }
