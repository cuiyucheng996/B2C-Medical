"""个人记忆 RAG 查看：切换 user_id，看卡片与检索结果。"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from configs.paths_b2c import ensure_b2c_runtime_path

ensure_b2c_runtime_path()

from server.web_ui import mount_b2c_web_ui
from server.builtin_server.memory_rag.logic import (
    build_timeline,
    list_recent_user_ids,
    load_card,
    rag_probe,
    search_memory,
)

_SECRET = (os.getenv("MEMORY_RAG_SECRET") or "b2c-memory-rag-dev-secret").strip()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    from infra.registry.lifecycle import registry_shutdown, registry_startup

    registry_startup("b2c-memory-rag")
    yield
    registry_shutdown()


app = FastAPI(title="memory_rag", version="0.1.0", lifespan=_lifespan)
from server.otel_app import mount_otel

mount_otel(app, "b2c-memory-rag")
app.add_middleware(SessionMiddleware, secret_key=_SECRET, session_cookie="memory_rag_session")
mount_b2c_web_ui(app, "memory.html")


@app.middleware("http")
async def _trace_middleware(request: Request, call_next):
    from server.trace_http import TRACE_HEADER, bind_trace_id, current_trace_id, inbound_trace_id

    tid = inbound_trace_id(dict(request.headers))
    if tid:
        bind_trace_id(tid)
    response = await call_next(request)
    out = current_trace_id()
    if out:
        response.headers[TRACE_HEADER] = out
    return response


def _uid(request: Request) -> str:
    session = request.session
    return str(session.get("user_id") or "").strip()


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "healthy", "service": "memory_rag"}


class LoginBody(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=120)


@app.post("/api/login")
def api_login(body: LoginBody, request: Request) -> dict[str, Any]:
    uid = body.user_id.strip()
    request.session["user_id"] = uid
    return {"ok": True, "user_id": uid}


@app.post("/api/logout")
def api_logout(request: Request) -> dict[str, Any]:
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def api_me(request: Request) -> dict[str, Any]:
    uid = _uid(request)
    return {"ok": bool(uid), "user_id": uid or None}


@app.get("/api/users/suggestions")
def api_user_suggestions() -> dict[str, Any]:
    return {"ok": True, "user_ids": list_recent_user_ids()}


@app.get("/api/card")
def api_card(request: Request, limit: int = 30) -> dict[str, Any]:
    uid = _uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="请先登录 user_id")
    result = load_card(uid, limit=min(max(limit, 5), 80))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/api/timeline")
def api_timeline(request: Request, limit: int = 40) -> dict[str, Any]:
    uid = _uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="请先登录 user_id")
    result = load_card(uid, limit=min(max(limit, 5), 80))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    card = result.get("card") if isinstance(result.get("card"), dict) else {}
    return {"ok": True, "user_id": uid, "timeline": build_timeline(card)}


class SearchBody(BaseModel):
    query: str = ""
    max_facts: int = 12


@app.post("/api/search")
def api_search(body: SearchBody, request: Request) -> dict[str, Any]:
    uid = _uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="请先登录 user_id")
    result = search_memory(uid, body.query, max_facts=body.max_facts)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result


class RagBody(BaseModel):
    query: str = Field(..., min_length=1)


@app.post("/api/rag")
def api_rag(body: RagBody, request: Request) -> dict[str, Any]:
    uid = _uid(request)
    if not uid:
        raise HTTPException(status_code=401, detail="请先登录 user_id")
    result = rag_probe(uid, body.query)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result)
    return result
