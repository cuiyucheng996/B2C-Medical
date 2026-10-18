"""定向采集入口：页面在本进程，处理转发到陪聊服务。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from server.upstream import request_json, request_multipart
from server.web_ui import mount_b2c_web_ui

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from infra.registry.lifecycle import registry_shutdown, registry_startup

    registry_startup("b2c-directed-crawl")
    yield
    registry_shutdown()


app = FastAPI(title="directed_crawl", version="0.1.0", lifespan=_lifespan)
from server.otel_app import mount_otel

mount_otel(app, "b2c-directed-crawl")
mount_b2c_web_ui(app, "directed_crawl.html")


def _raise(status: int, body: Any) -> dict[str, Any]:
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return body if isinstance(body, dict) else {"ok": False, "error": body}


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    from server.builtin_server.directed_crawl.yingdao_mcp import connection_ready, mcp_script_path

    return {
        "status": "healthy",
        "service": "directed_crawl",
        "mode": "proxy+rpa_local",
        "yingdao_mcp": str(mcp_script_path()),
        "yingdao_mcp_ready": str(connection_ready()).lower(),
    }


@app.get("/catalog")
def get_catalog() -> dict[str, Any]:
    status, body = request_json("GET", "/internal/crawl/catalog")
    return _raise(status, body)


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


@app.post("/fetch")
def post_fetch(body: FetchBody) -> dict[str, Any]:
    status, payload = request_json("POST", "/internal/crawl/fetch", body.model_dump())
    return _raise(status, payload)


class RpaStartBody(BaseModel):
    profile: str
    session_id: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


@app.post("/rpa/start")
def post_rpa_start(body: RpaStartBody) -> dict[str, Any]:
    from server.builtin_server.directed_crawl.logic import start_directed_rpa

    result = start_directed_rpa(
        profile=body.profile,
        params=body.params,
        session_id=body.session_id,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/pipeline/run")
def post_pipeline_run(body: PipelineRunBody) -> dict[str, Any]:
    status, payload = request_json(
        "POST",
        "/internal/crawl/pipeline/run",
        body.model_dump(),
    )
    return _raise(status, payload)


class WebhookBody(BaseModel):
    source: str = "rpa"
    status: str = "finished"
    job_type: str = ""
    crawl_date: str = ""
    excel_path: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


@app.post("/webhook/start-crawl")
def webhook_start_crawl(body: WebhookBody) -> dict[str, Any]:
    status, result = request_json("POST", "/internal/crawl/webhook/start-crawl", body.model_dump())
    return _raise(status, result)


@app.post("/ingest/excel")
async def ingest_excel(
    file: UploadFile = File(...),
    job_type: str = Form(""),
    source: str = Form("upload"),
) -> dict[str, Any]:
    status, body = request_multipart(
        "/internal/crawl/ingest/excel",
        file_field="file",
        filename=file.filename or "upload.xlsx",
        data=await file.read(),
        fields={"job_type": job_type, "source": source},
    )
    return _raise(status, body)


@app.get("/jobs/latest")
def jobs_latest() -> dict[str, Any]:
    status, body = request_json("GET", "/internal/crawl/jobs/latest")
    return _raise(status, body)


@app.get("/jobs/{job_id}")
def jobs_one(job_id: str) -> dict[str, Any]:
    status, body = request_json("GET", f"/internal/crawl/jobs/{job_id}")
    return _raise(status, body)


class ComposeBody(BaseModel):
    job_id: str = ""


@app.post("/compose")
def post_compose(body: ComposeBody) -> dict[str, Any]:
    status, payload = request_json("POST", "/internal/crawl/compose", body.model_dump())
    return _raise(status, payload)
