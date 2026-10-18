"""报告解读的真正处理，只由陪聊进程挂载。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from server.builtin_server.report_read.logic import compose_report, gate_report_file, lookup_report_handbook, pull_report_facts

router = APIRouter()


@router.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "service": "report_read"}


@router.post("/gate")
async def gate(
    file: UploadFile = File(...),
    source: str = Form("api"),
    user_id: str = Form("report-api"),
) -> dict[str, Any]:
    try:
        result = gate_report_file(
            filename=file.filename or "report.bin",
            data=await file.read(),
            source=source,
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail={"message": "不是合规的体检/化验报告", **result})
    return result


class FactsBody(BaseModel):
    last_file: dict[str, Any] = Field(default_factory=dict)
    layout: str = "v2"


class HandbookBody(BaseModel):
    query: str = ""


@router.post("/facts")
def facts(body: FactsBody) -> dict[str, Any]:
    return pull_report_facts(body.last_file, layout=body.layout)


@router.post("/handbook")
def handbook(body: HandbookBody) -> dict[str, Any]:
    return lookup_report_handbook(body.query)


class ComposeBody(BaseModel):
    gate: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    handbook: dict[str, Any] = Field(default_factory=dict)


@router.post("/compose")
def compose(body: ComposeBody) -> dict[str, Any]:
    return compose_report(gate=body.gate, facts=body.facts, handbook=body.handbook)
