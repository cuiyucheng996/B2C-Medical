"""定向采集的真正处理，只由陪聊进程挂载。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from server.builtin_server.directed_crawl.logic import (
    catalog,
    compose_companion_card,
    fetch_directed,
    get_job,
    get_latest_job,
    health_payload,
    ingest_excel_bytes,
    run_pipeline,
    start_crawl_webhook,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return health_payload()


@router.get("/catalog")
def get_catalog() -> dict[str, Any]:
    return catalog()


class FetchBody(BaseModel):
    url: str
    profile: str = "health_guide"
    session_id: str = ""


class PipelineStepBody(BaseModel):
    id: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineRunBody(BaseModel):
    url: str
    profile: str = "health_guide"
    session_id: str = ""
    pipeline: str = "page_fetch"
    steps: list[PipelineStepBody] | None = None
    dry_run: bool = False


@router.post("/fetch")
def post_fetch(body: FetchBody) -> dict[str, Any]:
    result = fetch_directed(url=body.url, profile=body.profile, session_id=body.session_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/pipeline/run")
def post_pipeline_run(body: PipelineRunBody) -> dict[str, Any]:
    steps = [s.model_dump() for s in body.steps] if body.steps else None
    result = run_pipeline(
        url=body.url,
        profile=body.profile,
        session_id=body.session_id,
        pipeline=body.pipeline,
        steps=steps,
        dry_run=body.dry_run,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


class WebhookBody(BaseModel):
    source: str = "rpa"
    status: str = "finished"
    job_type: str = ""
    crawl_date: str = ""
    excel_path: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


@router.post("/webhook/start-crawl")
def webhook_start_crawl(body: WebhookBody) -> dict[str, Any]:
    payload = body.model_dump()
    payload.update(body.extra or {})
    return start_crawl_webhook(payload)


@router.post("/ingest/excel")
async def ingest_excel(
    file: UploadFile = File(...),
    job_type: str = Form(""),
    source: str = Form("upload"),
) -> dict[str, Any]:
    result = ingest_excel_bytes(
        filename=file.filename or "upload.xlsx",
        data=await file.read(),
        job_type=job_type,
        source=source,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@router.get("/jobs/latest")
def jobs_latest() -> dict[str, Any]:
    result = get_latest_job()
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result)
    return result


@router.get("/jobs/{job_id}")
def jobs_one(job_id: str) -> dict[str, Any]:
    result = get_job(job_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result)
    return result


class ComposeBody(BaseModel):
    job_id: str = ""


@router.post("/compose")
def post_compose(body: ComposeBody) -> dict[str, Any]:
    job = get_job(body.job_id) if body.job_id else get_latest_job()
    if not job.get("ok"):
        raise HTTPException(status_code=404, detail=job)
    return {"ok": True, "job_id": job.get("job_id"), "card": compose_companion_card(job)}
