"""B2C 陪聊 API：对齐 Graphiti graph-service 的 ingest / retrieve 形态。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.companion_agent import CompanionAgent
from memory.graphiti_store import add_episode, persist_session_findings, search_facts
from session import get_session_store

logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "web-frontend" / "dist"


def _html_page(name: str) -> Path:
    fallback = WEB_DIR / name
    if fallback.is_file():
        return fallback
    built = FRONTEND_DIST / name
    if built.is_file():
        return built
    raise HTTPException(status_code=404, detail=f"{name} missing")


async def _digest_loop() -> None:
    from algorithm.periodic_digest import run_daily_digests

    await asyncio.sleep(3)
    while True:
        try:
            run_daily_digests()
        except Exception:
            logger.exception("daily digest failed")
        await asyncio.sleep(24 * 60 * 60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_digest_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="B2C Companion",
    description="长期陪聊 + Finding 记忆（非 LangGraph 问诊）",
    lifespan=lifespan,
)
_agent = CompanionAgent()
_sessions = get_session_store()


class SessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="对应 Graphiti group_id / Neo4j UserInfo.id")


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    resume_token: str = ""
    input_channel: Literal["text", "asr"] = "text"
    think: bool = False


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Message(BaseModel):
    content: str
    role_type: Literal["user", "assistant", "system"] = "user"
    role: str | None = None


class AddMessagesRequest(BaseModel):
    group_id: str
    messages: list[Message]


class SearchQuery(BaseModel):
    query: str
    group_ids: list[str] = Field(default_factory=list)
    max_facts: int = 8


class GetMemoryRequest(BaseModel):
    group_id: str
    messages: list[Message] = Field(default_factory=list)
    max_facts: int = 8


class ChunkConfigBody(BaseModel):
    embedding_provider: str | None = None
    embedding_model: str | None = None
    ragflow_embedding_model: str | None = None
    chunk_method: str | None = None
    chunk_token_num: int | None = None
    delimiter: str | None = None
    layout_recognize: str | None = None
    local_max_chars: int | None = None
    local_naive_chars: int | None = None
    local_overlap: int | None = None


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/tools")
def api_tool_registry() -> dict[str, Any]:
    """C 端工具注册表（本地 + 已登记的第三方 MCP）。"""
    from tools.registry import list_companion_registry
    from tools.setup import INTENT_BIND

    binds = [
        {"major": major, "handler": handler or "默认", "tools": list(names)}
        for (major, handler), names in INTENT_BIND.items()
    ]
    return {"tools": list_companion_registry(), "intent_bind": binds}


@app.get("/api/kb")
def api_kb_status() -> dict[str, Any]:
    """RAGFlow 文档库状态。分块请打开返回的 web_url。"""
    from algorithm.ragflow_kb import kb_status

    return kb_status()


@app.get("/api/kb/documents")
def api_kb_documents() -> dict[str, Any]:
    from algorithm.ragflow_kb import kb_status, list_documents

    status = kb_status()
    if not status.get("reachable"):
        raise HTTPException(status_code=503, detail=status.get("error") or "RAGFlow 不可用")
    return {"documents": list_documents(), "web_url": status.get("web_url")}


@app.post("/api/kb/documents")
async def api_kb_upload(file: UploadFile = File(...), replace: bool = True) -> dict[str, Any]:
    """动态更新公共知识 PDF：默认同名覆盖后重新分块。"""
    import tempfile

    from algorithm.ragflow_kb import replace_pdf

    filename = file.filename or "handbook.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    tmp = Path(tempfile.mkdtemp()) / Path(filename).name
    tmp.write_bytes(data)
    try:
        return replace_pdf(tmp, replace_same_name=replace)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)



class RegisterRequest(BaseModel):
    login_name: str = Field(..., min_length=1)
    display_name: str = ""
    pin: str = ""


class LoginRequest(BaseModel):
    login_name: str = Field(..., min_length=1)
    pin: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_html_page("login.html"))


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(_html_page("companion.html"))


@app.get("/tools")
def tools_page() -> FileResponse:
    return FileResponse(_html_page("tools.html"))


@app.get("/eval")
def eval_page() -> FileResponse:
    return FileResponse(_html_page("eval.html"))


class EvalRunRequest(BaseModel):
    extra_texts: list[str] = Field(default_factory=list)


@app.get("/api/eval/env")
def api_eval_env() -> dict[str, Any]:
    from algorithm.eval_suite import probe_env

    return probe_env()


@app.post("/api/eval/run")
def api_eval_run(req: EvalRunRequest) -> dict[str, Any]:
    from algorithm.eval_suite import run_eval

    extras = [str(t).strip() for t in req.extra_texts if str(t).strip()]
    return run_eval(extra_texts=extras[:20])


@app.post("/api/eval/one")
def api_eval_one(req: ChatRequest) -> dict[str, Any]:
    from algorithm.eval_suite import classify_one

    row = classify_one(req.text)
    if not row.get("ok"):
        raise HTTPException(status_code=400, detail=row.get("reason") or "empty")
    return row


class CorefEvalRequest(BaseModel):
    query: str = ""
    prior: str = ""
    slot_memory: dict[str, list[str]] = Field(default_factory=dict)
    gold: bool = False


class KbEvalRequest(BaseModel):
    query: str = ""
    extra: str = ""
    top_k: int = 8


class KbToolProbeRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tools: list[str] = Field(default_factory=list)
    handler: str = ""
    rag_enhanced: bool | None = None


class RagEnhanceConfigBody(BaseModel):
    enabled: bool | None = None
    multi_query: bool | None = None
    max_queries: int | None = None
    rrf_k: int | None = None
    fetch_k: int | None = None
    top_k: int | None = None
    use_cache: bool | None = None
    cache_ttl_seconds: int | None = None
    keyword_rerank: bool | None = None


@app.post("/api/eval/coref")
def api_eval_coref(req: CorefEvalRequest) -> dict[str, Any]:
    from algorithm.coref_eval import run_coref_eval, run_coref_one

    if req.gold or not (req.query or "").strip():
        return run_coref_eval()
    row = run_coref_one(query=req.query, prior=req.prior, slot_memory=req.slot_memory)
    if not row.get("ok"):
        raise HTTPException(status_code=400, detail=row.get("reason") or "empty")
    return row


@app.get("/api/eval/kb/tools")
def api_eval_kb_tools_catalog() -> dict[str, Any]:
    from algorithm.kb_eval import list_kb_probe_handlers, list_kb_probe_tools

    return {"tools": list_kb_probe_tools(), "handlers": list_kb_probe_handlers()}


@app.post("/api/eval/kb")
def api_eval_kb(req: KbEvalRequest) -> dict[str, Any]:
    from algorithm.kb_eval import run_kb_eval

    text = (req.query or req.extra or "").strip()
    try:
        return run_kb_eval(query=text, top_k=req.top_k)
    except Exception as exc:
        logger.exception("eval kb failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/eval/kb/tools")
def api_eval_kb_tool_probe(req: KbToolProbeRequest) -> dict[str, Any]:
    from algorithm.kb_eval import run_kb_tool_probe

    try:
        row = run_kb_tool_probe(
            query=req.query,
            tools=req.tools,
            handler=req.handler,
            rag_enhanced=req.rag_enhanced,
        )
    except Exception as exc:
        logger.exception("eval kb tool probe failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if row.get("reason") == "empty":
        raise HTTPException(status_code=400, detail="empty")
    if row.get("reason") == "no_tools":
        raise HTTPException(status_code=400, detail="请至少选择一个工具")
    return row


@app.post("/api/eval/file")
async def api_eval_file(
    file: UploadFile = File(...),
    table_engine: str = "gate",
    us_strategy: str = "hi_res",
    us_infer_table: bool = True,
    us_languages: str = "eng,chi_sim",
    gate_lab_threshold: int = 2,
    gate_run_explain: bool = True,
    gate_layout_only: bool = True,
) -> dict[str, Any]:
    from algorithm.file_eval import run_file_eval

    filename = file.filename or "upload.bin"
    data = await file.read()
    try:
        return run_file_eval(
            filename=filename,
            data=data,
            table_engine=table_engine,
            us_strategy=us_strategy,
            us_infer_table=us_infer_table,
            us_languages=us_languages,
            gate_lab_threshold=gate_lab_threshold,
            gate_run_explain=gate_run_explain,
            gate_layout_only=gate_layout_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("eval file failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eval/rag-config")
def api_eval_rag_config() -> dict[str, Any]:
    from algorithm.rag.config import get_rag_config

    cfg = get_rag_config()
    return {
        "enabled": cfg.enhanced,
        "multi_query": cfg.multi_query,
        "max_queries": cfg.max_queries,
        "rrf_k": cfg.rrf_k,
        "fetch_k": cfg.fetch_k,
        "top_k": cfg.top_k,
        "use_cache": cfg.use_cache,
        "cache_ttl_seconds": cfg.cache_ttl_seconds,
        "keyword_rerank": cfg.keyword_rerank,
    }


@app.put("/api/eval/rag-config")
def api_eval_rag_config_put(req: RagEnhanceConfigBody) -> dict[str, Any]:
    from algorithm.rag.config import set_rag_config

    return set_rag_config(req.model_dump(exclude_none=True))


@app.get("/api/eval/chunk-config")
def api_eval_chunk_config() -> dict[str, Any]:
    from algorithm.chunk_config import get_chunk_config

    return get_chunk_config()


@app.put("/api/eval/chunk-config")
def api_eval_chunk_config_put(req: ChunkConfigBody) -> dict[str, Any]:
    from algorithm.chunk_config import set_chunk_config

    try:
        return set_chunk_config(req.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/eval/ragflow-ingest")
async def api_eval_ragflow_ingest(
    file: UploadFile = File(...),
    chunk_source: str = "local",
    skip_ragflow: bool = False,
    chunk_token_num: int | None = None,
    local_max_chars: int | None = None,
    local_naive_chars: int | None = None,
    local_overlap: int | None = None,
    embedding_model: str | None = None,
    ragflow_embedding_model: str | None = None,
) -> dict[str, Any]:
    from algorithm.chunk_config import get_chunk_config, set_chunk_config
    from algorithm.ragflow_eval import run_ragflow_ingest_eval

    patch = {
        "chunk_token_num": chunk_token_num,
        "local_max_chars": local_max_chars,
        "local_naive_chars": local_naive_chars,
        "local_overlap": local_overlap,
        "embedding_model": embedding_model,
        "ragflow_embedding_model": ragflow_embedding_model,
    }
    if any(value is not None and value != "" for value in patch.values()):
        try:
            set_chunk_config(patch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    filename = file.filename or "upload.pdf"
    data = await file.read()
    try:
        row = run_ragflow_ingest_eval(
            filename=filename,
            data=data,
            chunk_source=chunk_source,
            skip_ragflow=skip_ragflow,
        )
        row["chunk_config"] = get_chunk_config()
        return row
    except Exception as exc:
        logger.exception("eval ragflow ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/users")
def api_list_users() -> dict[str, Any]:
    from algorithm.user_accounts import list_users

    try:
        return {"ok": True, "users": list_users()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 UserInfo 失败: {exc}") from exc


@app.post("/api/users")
def api_create_user(req: RegisterRequest) -> dict[str, Any]:
    from algorithm.user_accounts import create_user

    result = create_user(login_name=req.login_name, display_name=req.display_name, pin=req.pin)
    if not result.get("ok"):
        reason = str(result.get("reason") or "create_failed")
        status = 409 if reason == "login_name_taken" else 400
        raise HTTPException(status_code=status, detail=reason)
    return result


@app.post("/api/auth/login")
def api_login(req: LoginRequest) -> dict[str, Any]:
    from algorithm.user_accounts import login_user

    result = login_user(login_name=req.login_name, pin=req.pin)
    if not result.get("ok"):
        reason = str(result.get("reason") or "login_failed")
        status = 404 if reason == "not_found" else 401 if reason == "bad_pin" else 400
        raise HTTPException(status_code=status, detail=reason)
    return result


@app.post("/api/digests/run")
def api_run_digests() -> dict[str, Any]:
    """手动跑一遍每日小结。服务启动约 3 秒后也会自动跑，之后每 24 小时一次。"""
    from algorithm.periodic_digest import run_daily_digests

    try:
        return run_daily_digests()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sessions")
def create_session(req: SessionCreateRequest) -> dict[str, Any]:
    from algorithm.user_accounts import resolve_user_ref

    try:
        session = _sessions.create(resolve_user_ref(req.user_id.strip()) or req.user_id.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from algorithm.greeting_push import build_entry_push

    push = build_entry_push(session.group_id)
    session.append(role_type="assistant", content=str(push.get("text") or ""), role="assistant")
    payload = session.to_dict()
    payload["push"] = push
    return payload


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session.to_dict()


@app.post("/api/sessions/{session_id}/close")
def close_session(session_id: str) -> dict[str, Any]:
    """关页 / 点结束：把本场用户话抽离入库。过程中不写 Neo4j。"""
    session = _sessions.get(session_id)
    if session is None:
        return {
            "session": None,
            "ingest": {"ok": True, "skipped": True, "reason": "already_gone"},
        }
    ingest = persist_session_findings(session)
    _sessions.drop(session)
    return {"session": session.to_dict(), "ingest": ingest}


@app.post("/api/sessions/{session_id}/report")
async def upload_report(
    session_id: str,
    file: UploadFile = File(...),
    think: bool = Form(False),
) -> dict[str, Any]:
    """上传图片或 PDF：先判形态和内容，化验单走报告解读，普通图/文档走内容说明。"""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    filename = file.filename or "report.pdf"
    data = await file.read()
    from algorithm.file_gate import ingest_user_file

    try:
        ingested = ingest_user_file(user_id=session.group_id, filename=filename, data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("report pdf parse failed")
        raise HTTPException(status_code=500, detail=f"解析失败: {exc}") from exc
    session.last_file = ingested.to_dict()
    try:
        payload = await _agent.areply(session, ingested.user_text, think=think)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload["file"] = session.last_file
    payload["report"] = session.last_report or None
    return payload


@app.get("/api/voice/status")
def voice_status() -> dict[str, Any]:
    from algorithm.voice_client import voice_status as status

    return status()


@app.post("/api/voice/asr")
async def voice_asr(file: UploadFile = File(...)) -> dict[str, str]:
    from algorithm.voice_client import transcribe_audio

    data = await file.read()
    try:
        text = transcribe_audio(data, file.filename or "audio.webm", file.content_type or "audio/webm")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"text": text}


@app.post("/api/voice/tts")
async def voice_tts(req: TtsRequest) -> Response:
    from algorithm.voice_client import synthesize_speech

    try:
        audio, mime = synthesize_speech(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(content=audio, media_type=mime)


@app.post("/api/sessions/{session_id}/chat")
def chat(session_id: str, req: ChatRequest) -> dict[str, Any]:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    try:
        return _agent.reply(
            session, req.text, resume_token=req.resume_token, think=req.think
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/chat/stream")
async def chat_stream(session_id: str, req: ChatRequest) -> StreamingResponse:
    """对齐主项目 /chat/stream：SSE 推 thinking / token / state / done。"""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    async def event_gen():
        try:
            async for chunk in _agent.stream_reply(
                session, req.text, resume_token=req.resume_token, think=req.think
            ):
                yield chunk
        except ValueError as exc:
            from configs.paths_b2c import ensure_uvgraph_root

            ensure_uvgraph_root()
            from src.langchain.agent.stream_agent import sse_payload

            yield sse_payload({"event": "error", "message": str(exc)})
            yield sse_payload({"event": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/messages")
def add_messages(req: AddMessagesRequest) -> dict[str, Any]:
    """对齐 Graphiti POST /messages：把用户话当 Episode 入库。"""
    ingested = 0
    for msg in req.messages:
        result = add_episode(
            user_id=req.group_id,
            content=msg.content,
            role_type=msg.role_type,
        )
        if result.get("ok"):
            ingested += 1
    return {"success": True, "message": "queued", "ingested": ingested}


@app.post("/search")
def search(req: SearchQuery) -> dict[str, Any]:
    group_id = (req.group_ids[0] if req.group_ids else "").strip()
    if not group_id:
        raise HTTPException(status_code=400, detail="group_ids 不能为空")
    facts = search_facts(user_id=group_id, query=req.query, max_facts=req.max_facts)
    return {"facts": facts}


@app.post("/get-memory")
def get_memory(req: GetMemoryRequest) -> dict[str, Any]:
    query = " ".join(m.content for m in req.messages) or "近况"
    facts = search_facts(user_id=req.group_id, query=query, max_facts=req.max_facts)
    return {"facts": facts}


_ASSETS_DIR = FRONTEND_DIST / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="companion-assets")
