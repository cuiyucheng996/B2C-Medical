"""文档入库入口：页面在本进程，处理转发到陪聊服务。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from server.upstream import request_multipart
from server.web_ui import mount_b2c_web_ui

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from infra.registry.lifecycle import registry_shutdown, registry_startup

    registry_startup("b2c-doc-ingest")
    yield
    registry_shutdown()


app = FastAPI(title="doc_ingest", version="0.1.0", lifespan=_lifespan)
from server.otel_app import mount_otel

mount_otel(app, "b2c-doc-ingest")
mount_b2c_web_ui(app, "doc_ingest.html")


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "service": "doc_ingest", "mode": "proxy"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    require_bold: str = Form("true"),
    max_lines: int = Form(10),
    max_chars: int = Form(80),
    heading_pattern: str = Form(r"^\d{1,3}\.\s*\S"),
) -> dict[str, Any]:
    status, body = request_multipart(
        "/internal/doc/ingest",
        file_field="file",
        filename=file.filename or "upload.bin",
        data=await file.read(),
        fields={
            "require_bold": require_bold,
            "max_lines": str(max_lines),
            "max_chars": str(max_chars),
            "heading_pattern": heading_pattern,
        },
    )
    if status >= 400:
        raise HTTPException(status_code=status, detail=body)
    return body if isinstance(body, dict) else {"ok": False, "error": body}
