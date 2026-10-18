"""文档入库的真正处理，只由陪聊进程挂载。"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from server.builtin_server.doc_ingest.logic import health_payload, ingest_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return health_payload()


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    require_bold: str = Form("true"),
    max_lines: int = Form(10),
    max_chars: int = Form(80),
    heading_pattern: str = Form(r"^\d{1,3}\.\s*\S"),
) -> dict[str, Any]:
    data = await file.read()
    from server.builtin_server.jobs.queue import async_heavy_enabled, enqueue

    if async_heavy_enabled():
        from configs.paths_b2c import B2C_ROOT
        from uuid import uuid4

        folder = B2C_ROOT / ".cache" / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        name = file.filename or "upload.bin"
        path = folder / f"{uuid4().hex}_{name}"
        path.write_bytes(data)
        job_id = enqueue(
            "doc_ingest",
            {
                "path": str(path),
                "filename": name,
                "heading": {
                    "require_bold": str(require_bold).strip().lower() in {"1", "true", "yes", "on"},
                    "max_lines": max_lines,
                    "max_chars": max_chars,
                    "pattern": heading_pattern,
                },
            },
        )
        return {"ok": True, "deferred": True, "job_id": job_id, "path": str(path)}
    try:
        result = ingest_document(
            filename=file.filename or "upload.bin",
            data=data,
            heading={
                "require_bold": str(require_bold).strip().lower() in {"1", "true", "yes", "on"},
                "max_lines": max_lines,
                "max_chars": max_chars,
                "pattern": heading_pattern,
            },
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("doc_ingest /ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result
