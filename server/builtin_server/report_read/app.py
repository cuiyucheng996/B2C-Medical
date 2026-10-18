"""报告解读入口：页面在本进程，处理转发到陪聊服务。"""

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

    registry_startup("b2c-report-read")
    yield
    registry_shutdown()


app = FastAPI(title="report_read", version="0.1.0", lifespan=_lifespan)
from server.otel_app import mount_otel

mount_otel(app, "b2c-report-read")
mount_b2c_web_ui(app, "report_read.html")


def _raise(status: int, body: Any) -> dict[str, Any]:
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return body if isinstance(body, dict) else {"ok": False, "error": body}


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "service": "report_read", "mode": "proxy"}


@app.post("/gate")
async def gate(
    file: UploadFile = File(...),
    source: str = Form("api"),
    user_id: str = Form("report-api"),
) -> dict[str, Any]:
    status, body = request_multipart(
        "/internal/report/gate",
        file_field="file",
        filename=file.filename or "report.bin",
        data=await file.read(),
        fields={"source": source, "user_id": user_id},
    )
    return _raise(status, body)


class FactsBody(BaseModel):
    last_file: dict[str, Any] = Field(default_factory=dict)
    layout: str = "v2"


class HandbookBody(BaseModel):
    query: str = ""


@app.post("/facts")
def facts(body: FactsBody) -> dict[str, Any]:
    status, payload = request_json("POST", "/internal/report/facts", body.model_dump())
    return _raise(status, payload)


@app.post("/handbook")
def handbook(body: HandbookBody) -> dict[str, Any]:
    status, payload = request_json("POST", "/internal/report/handbook", body.model_dump())
    return _raise(status, payload)


class ComposeBody(BaseModel):
    gate: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    handbook: dict[str, Any] = Field(default_factory=dict)


@app.post("/compose")
def compose(body: ComposeBody) -> dict[str, Any]:
    status, payload = request_json("POST", "/internal/report/compose", body.model_dump())
    return _raise(status, payload)
