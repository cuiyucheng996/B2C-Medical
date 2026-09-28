"""B2C 陪聊：多意图可并行 create_agent，再拼回复。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from algorithm.chat_preprocess import (
    finish_preprocess_after_intent,
    format_preprocess_for_agent,
    run_chat_preprocess,
)
from algorithm.companion_turn import CompanionTurnContext, attach_intent_to_preprocess
from algorithm.intent_classify import intent_result_to_dict
from algorithm.reply_stream import ReplyTextExtractor, tool_call_name
from algorithm.short_memory import (
    load_b2c_agent_history,
    load_b2c_short_messages,
    save_b2c_slot_memory,
    sync_b2c_session_memory,
    sync_b2c_short_memory,
    touch_b2c_session_redis,
)
from agent.agent_checkpointer import get_agent_checkpointer
from graph.checkpointer import companion_invoke_config
from middleware import companion_middleware
from middleware.hitl_config import companion_tool_hitl_enabled
from middleware.context import CompanionRunContext
from middleware.tool_pairs import clear_agent_checkpoint_thread, drop_orphan_tool_messages
from middleware.middleware_trace import reset_middleware_trace
from middleware.tool_trace import reset_tool_trace, take_tool_trace
from algorithm.llm_trace import llm_trace_to_briefs, reset_llm_trace, take_llm_trace
from schema.companion_reply import CompanionReply
from session import CompanionSession
from tools.companion_memory import companion_last_facts, companion_parsed_report, companion_user_id
from tools.companion_tools import companion_turn, tools_for_intent

logger = logging.getLogger(__name__)

__all__ = ["CompanionAgent", "create_companion_agent"]

_last_graph_trace: ContextVar[dict[str, Any]] = ContextVar("companion_graph_trace", default={})
_companion_tool_hitl_row: ContextVar[dict[str, Any] | None] = ContextVar(
    "companion_tool_hitl_row",
    default=None,
)


def create_companion_agent(
    model: Any,
    tools: list[BaseTool] | None = None,
    *,
    enable_tool_hitl: bool | None = None,
) -> Any:
    """本轮工具集不同；提示词、观察、MCP 护栏、信封、重试均走中间件。

    有没有业务工具，最终一句都用 CompanionReply。页面只收这句。
    """
    bound = list(tools or [])
    use_hitl = enable_tool_hitl if enable_tool_hitl is not None else bool(bound)
    checkpointer = get_agent_checkpointer() if use_hitl and companion_tool_hitl_enabled() else None
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": bound,
        "middleware": companion_middleware(enable_tool_hitl=use_hitl),
        "context_schema": CompanionRunContext,
        "name": "companion_chat",
        "response_format": ToolStrategy(CompanionReply),
        "checkpointer": checkpointer,
    }
    return create_agent(**kwargs)


def _agent_thread_id(turn: dict[str, Any] | None, task: dict[str, Any] | None) -> str:
    sid = str((turn or {}).get("session_id") or "").strip()
    major = str((task or {}).get("major") or "").strip()
    handler = str((task or {}).get("handler") or (turn or {}).get("handler") or "").strip()
    if sid:
        return f"agent:{sid}:{major}:{handler}"
    return "agent:anonymous"


def _hitl_request_from_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    intr = result.get("__interrupt__")
    if not intr:
        return None
    first = intr[0]
    value = getattr(first, "value", None)
    return value if isinstance(value, dict) else None


def _chat_model() -> Any:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.llm.chat import get_chat_model

    return get_chat_model()


def _clean_reply_leading_none(text: str) -> str:
    s = str(text or "")
    while s.upper().startswith("NONE"):
        s = s[4:]
    return s.strip()


def _reply_from_agent_result(result: dict[str, Any], *, streamed: str = "") -> str:
    structured = result.get("structured_response")
    if isinstance(structured, CompanionReply):
        text = (structured.assistant_reply or "").strip()
        if text:
            return _clean_reply_leading_none(text)
    if isinstance(structured, dict):
        text = str(structured.get("assistant_reply") or "").strip()
        if text:
            return _clean_reply_leading_none(text)
    if streamed.strip():
        # 结构化结果没拿到时，用已经流出去的那段，避免前端看到的内容被改写
        return _clean_reply_leading_none(streamed.strip())
    for msg in reversed(list(result.get("messages") or [])):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = str(getattr(msg, "content", "") or "").strip()
        if text:
            return _clean_reply_leading_none(text)
    return "我在，你可以继续说说近况。"


def _is_tool_role_400(exc: BaseException) -> bool:
    text = str(exc)
    return "tool_calls" in text and "role" in text and "tool" in text


async def _ainvoke_companion_agent(
    model: Any,
    tools: list[BaseTool],
    history: list[Any],
    observe: str,
    turn: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    on_token: Any = None,
    *,
    hitl_resume: dict[str, Any] | None = None,
    agent_thread_id: str = "",
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    turn_tok = None
    facts_tok = companion_last_facts.set([])
    if turn:
        turn_tok = companion_turn.set(turn)
    result: Any = None
    streamed = ""
    row = task or {}
    from algorithm.run_trace import bind_trace_id, current_trace_id

    tid = str((turn or {}).get("trace_id") or current_trace_id() or "").strip()
    if tid:
        bind_trace_id(tid)
    ctx = CompanionRunContext(
        query=str((turn or {}).get("query") or ""),
        observe=observe,
        major=str(row.get("major") or ""),
        intent=str(row.get("intent") or ""),
        handler=str(row.get("handler") or (turn or {}).get("handler") or ""),
        user_id=str((turn or {}).get("user_id") or ""),
        session_id=str((turn or {}).get("session_id") or ""),
        trace_id=tid,
        think_enabled=bool((turn or {}).get("think")),
        tool_trace=[],
    )
    human = ctx.query or "请根据预处理观察调用工具并回复。"
    tid = (agent_thread_id or "").strip() or _agent_thread_id(turn, row)
    invoke_config = companion_invoke_config(tid)
    hitl_tok = _companion_tool_hitl_row.set(None)
    reset_tool_trace()
    reset_middleware_trace()
    try:
        from langgraph.types import Command

        agent = create_companion_agent(model, tools)
        if hitl_resume:
            payload: Any = Command(resume=hitl_resume)
        else:
            # 每轮以 Redis short 的 history 为权威；checkpointer 会追加 messages，须先清线程避免重复进上下文
            clear_agent_checkpoint_thread(tid)
            payload = {
                "messages": drop_orphan_tool_messages([*history, HumanMessage(content=human)])
            }

        async def _run_once() -> tuple[Any, str]:
            if on_token is None or hitl_resume:
                out = await agent.ainvoke(payload, context=ctx, config=invoke_config)
                return out, ""
            return await _astream_companion_agent(
                agent, payload, ctx, on_token, config=invoke_config, bound_tools=bool(tools)
            )

        try:
            result, streamed = await _run_once()
        except Exception as exc:
            if _is_tool_role_400(exc) and tid and not hitl_resume:
                clear_agent_checkpoint_thread(tid)
                result, streamed = await _run_once()
            else:
                raise
    except Exception as exc:
        logger.exception("companion create_agent invoke failed")
        result = None
        if _is_tool_role_400(exc):
            try:
                fallback = create_companion_agent(model, [])
                packed = {
                    "messages": [
                        HumanMessage(content=(observe + "\n\n" + human).strip())
                    ]
                }
                if on_token is None:
                    result = await fallback.ainvoke(packed, context=ctx)
                else:
                    result, streamed = await _astream_companion_agent(
                        fallback, packed, ctx, on_token, bound_tools=False
                    )
            except Exception:
                logger.exception("companion invoke fallback without tools failed")
                result = None
    finally:
        facts = list(companion_last_facts.get() or [])
        companion_last_facts.reset(facts_tok)
        if turn_tok is not None:
            companion_turn.reset(turn_tok)
    if result is None:
        _companion_tool_hitl_row.reset(hitl_tok)
        return "我先记下了，网络波动稍后再聊也行。", facts, list(ctx.tool_trace or take_tool_trace())
    hitl_req = _hitl_request_from_result(result)
    if hitl_req and not hitl_resume:
        from algorithm.companion_hitl import (
            HITL_KIND_TOOL,
            format_tool_hitl_question,
            save_hitl,
        )

        question = format_tool_hitl_question(hitl_req)
        task_snap = {
            "major": row.get("major"),
            "intent": row.get("intent"),
            "handler": row.get("handler"),
            "observe": observe,
        }
        row_saved = save_hitl(
            str((turn or {}).get("session_id") or ""),
            question=question,
            agent_name="tool_hitl",
            hitl_kind=HITL_KIND_TOOL,
            state={
                "turn": dict(turn or {}),
                "task": task_snap,
                "observe": observe,
            },
            action_requests=list(hitl_req.get("action_requests") or []),
            review_configs=list(hitl_req.get("review_configs") or []),
            thread_id=tid,
            user_id=str((turn or {}).get("user_id") or ""),
        )
        _companion_tool_hitl_row.set(row_saved)
        return question, facts, list(ctx.tool_trace or take_tool_trace())
    _companion_tool_hitl_row.reset(hitl_tok)
    trace = list(ctx.tool_trace or [])
    if not trace and isinstance(result, dict):
        from middleware.tool_trace import tool_trace_from_messages

        trace = tool_trace_from_messages(list(result.get("messages") or []))
    if not trace:
        trace = take_tool_trace()
    if not isinstance(result, dict):
        return "我在，你可以继续说说近况。", facts, trace
    return _reply_from_agent_result(result, streamed=streamed), facts, trace


def _pop_tool_hitl_row() -> dict[str, Any] | None:
    row = _companion_tool_hitl_row.get()
    if row:
        _companion_tool_hitl_row.set(None)
    return row


def _hitl_response_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "pending": True,
        "hitl_kind": row.get("hitl_kind"),
        "resume_token": row.get("resume_token"),
        "question": row.get("question"),
        "agent_name": row.get("agent_name"),
        "action_requests": list(row.get("action_requests") or []),
        "review_configs": list(row.get("review_configs") or []),
    }


def _is_stream_garbage_token(text: str) -> bool:
    chunk = (text or "").strip()
    if not chunk:
        return True
    upper = chunk.upper()
    if upper == "NONE" or upper.replace("NONE", "") == "":
        return True
    return False


async def _astream_companion_agent(
    agent: Any,
    payload: dict[str, Any],
    ctx: CompanionRunContext,
    on_token: Any,
    *,
    config: dict[str, Any] | None = None,
    bound_tools: bool = False,
) -> tuple[Any, str]:
    """流式跑 agent：只把 CompanionReply 的 assistant_reply 增量吐给 on_token。

    业务工具参数和模型正文都不推。最终一句只从 CompanionReply 解出来。
    """
    del bound_tools
    extractor = ReplyTextExtractor()
    result: Any = None
    in_reply = False

    def _emit_piece(text: str) -> None:
        piece = _clean_reply_leading_none(text)
        if piece and not _is_stream_garbage_token(piece):
            on_token(piece)

    async for event in agent.astream_events(
        payload, context=ctx, version="v2", config=config or {}
    ):
        kind = event.get("event")
        if kind == "on_chat_model_start":
            extractor.reset()
            in_reply = False
            continue
        if kind in {"on_tool_start", "on_tool_end"}:
            continue
        if kind != "on_chat_model_stream":
            if kind == "on_chain_end" and event.get("name") == "companion_chat":
                data = (event.get("data") or {}).get("output")
                if isinstance(data, dict):
                    result = data
            continue
        chunk = (event.get("data") or {}).get("chunk")
        tname = tool_call_name(chunk)
        if tname == "CompanionReply":
            in_reply = True
        elif tname:
            in_reply = False
            from algorithm.stream_debug import emit_trace_tool_call_delta

            route = " / ".join(p for p in (ctx.major, ctx.intent, ctx.handler) if p)
            for item in getattr(chunk, "tool_call_chunks", None) or []:
                args_delta = str((item or {}).get("args") or "")
                if args_delta:
                    emit_trace_tool_call_delta(tname, args_delta, route=route)
            continue
        if not in_reply:
            continue
        for item in getattr(chunk, "tool_call_chunks", None) or []:
            delta = extractor.feed_args(str((item or {}).get("args") or ""))
            if delta:
                _emit_piece(delta)
                await asyncio.sleep(0)
    return result, _clean_reply_leading_none(extractor.text)


async def _ainvoke_intent_tasks(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    from graph.companion import ainvoke_companion_intent_graph

    reply, facts, trace = await ainvoke_companion_intent_graph(
        model, tasks, history, base_turn, on_token=on_token
    )
    _last_graph_trace.set(trace)
    return reply, facts


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("sync companion reply cannot run inside a running event loop")


class CompanionAgent:
    async def _resume_tool_hitl_if_any(
        self,
        session: CompanionSession,
        user_text: str,
        resume_token: str,
        hitl_decision: dict[str, Any] | None,
        *,
        think: bool = False,
    ) -> dict[str, Any] | None:
        from algorithm.companion_hitl import (
            HITL_KIND_TOOL,
            build_tool_hitl_resume,
            consume_hitl,
            get_pending_hitl,
        )
        from tools.companion_tools import tools_for_intent

        if not (resume_token or hitl_decision):
            return None
        pending = get_pending_hitl(session.session_id)
        if not pending or str(pending.get("hitl_kind") or "") != HITL_KIND_TOOL:
            return None
        pending = consume_hitl(
            session.session_id,
            resume_token,
            kinds=(HITL_KIND_TOOL,),
        )
        if not pending:
            return None
        state = pending.get("state") if isinstance(pending.get("state"), dict) else {}
        task_snap = state.get("task") if isinstance(state.get("task"), dict) else {}
        turn = state.get("turn") if isinstance(state.get("turn"), dict) else {}
        turn = {**turn, "think": think}
        observe = str(state.get("observe") or task_snap.get("observe") or "")
        model = _chat_model()
        if model is None:
            return {
                "reply": "已记下。当前未配置聊天模型。",
                "facts": [],
                "ingest": {"deferred": True},
                "preprocess": session.last_preprocess,
                "session": session.to_dict(),
                "hitl": _hitl_response_from_row(pending),
            }
        tools = tools_for_intent(
            major=str(task_snap.get("major") or ""),
            handler=str(task_snap.get("handler") or ""),
        )
        history = self._history_messages(session)
        resume_payload = build_tool_hitl_resume(
            pending,
            decision=hitl_decision,
            user_text=user_text,
        )
        gid = companion_user_id.set(session.group_id)
        try:
            reply, facts, _tool_trace = await _ainvoke_companion_agent(
                model,
                tools,
                history,
                observe,
                turn,
                task_snap,
                hitl_resume=resume_payload,
                agent_thread_id=str(pending.get("thread_id") or ""),
            )
        finally:
            companion_user_id.reset(gid)
        from algorithm.hitl_checkpoint import clear_tool_pending

        tool_row = _pop_tool_hitl_row()
        if tool_row:
            hitl = _hitl_response_from_row(tool_row)
            reply = str((hitl or {}).get("question") or reply)
        else:
            clear_tool_pending(session.session_id)
            hitl = None
        meta = {
            "preprocess": session.last_preprocess or {},
            "bound_tools": [str(getattr(t, "name", "") or "") for t in tools],
        }
        if user_text.strip():
            session.append(role_type="user", content=user_text.strip())
        return self._finish_turn(session, user_text or "（工具确认）", reply, meta, facts, hitl=hitl)

    def reply(
        self,
        session: CompanionSession,
        user_text: str,
        resume_token: str = "",
        think: bool = False,
        hitl_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resumed = _run_sync(
            self._resume_tool_hitl_if_any(
                session, user_text, resume_token, hitl_decision, think=think
            )
        )
        if resumed is not None:
            return resumed
        text, history, prep, skip_ask, hitl_state = self._begin_turn(
            session, user_text, resume_token=resume_token
        )
        gid = companion_user_id.set(session.group_id)
        facts_tok = companion_last_facts.set([])
        parsed_tok = companion_parsed_report.set(None)
        meta = {
            "search_query": "",
            "skipped_long_memory": True,
            "preprocess": prep.to_dict(),
            "bound_tools": [],
        }
        observe = format_preprocess_for_agent(prep)
        reply = "已记下。当前未配置聊天模型。"
        facts: list[dict[str, Any]] = []
        hitl: dict[str, Any] | None = None
        try:
            model = _chat_model()
            if model is None:
                reply = "已记下。当前未配置聊天模型。"
            else:
                shortcut = self._faq_shortcut_reply(model, text, history, prep.to_dict())
                if shortcut is not None:
                    reply, meta = shortcut
                else:
                    text, history, observe, _tools, meta, tasks = self._attach_intent(
                        session, text, history, prep, model
                    )
                    hitl = self._maybe_suspend(session, text, meta, skip_ask)
                    if hitl:
                        reply = str(hitl.get("question") or "")
                    else:
                        base_turn = self._turn_payload(
                            session, meta, text, hitl_state=hitl_state, think=think
                        )
                        reply, facts = _run_sync(
                            _ainvoke_intent_tasks(model, tasks, history, base_turn)
                        )
                    meta["langgraph"] = dict(_last_graph_trace.get() or {})
                    tool_row = _pop_tool_hitl_row()
                    if tool_row:
                        hitl = _hitl_response_from_row(tool_row)
                        reply = str((hitl or {}).get("question") or reply)
        finally:
            self._absorb_parsed_report(session)
            companion_user_id.reset(gid)
            companion_last_facts.reset(facts_tok)
            companion_parsed_report.reset(parsed_tok)
        return self._finish_turn(session, text, reply, meta, facts, hitl=hitl)

    async def areply(
        self,
        session: CompanionSession,
        user_text: str,
        resume_token: str = "",
        think: bool = False,
        hitl_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """已有事件循环里用：不要走 reply() 的 asyncio.run。"""
        resumed = await self._resume_tool_hitl_if_any(
            session, user_text, resume_token, hitl_decision, think=think
        )
        if resumed is not None:
            return resumed
        text, history, prep, skip_ask, hitl_state = self._begin_turn(
            session, user_text, resume_token=resume_token
        )
        gid = companion_user_id.set(session.group_id)
        facts_tok = companion_last_facts.set([])
        parsed_tok = companion_parsed_report.set(None)
        meta = {
            "search_query": "",
            "skipped_long_memory": True,
            "preprocess": prep.to_dict(),
            "bound_tools": [],
        }
        observe = format_preprocess_for_agent(prep)
        reply = "已记下。当前未配置聊天模型。"
        facts: list[dict[str, Any]] = []
        hitl: dict[str, Any] | None = None
        try:
            model = _chat_model()
            if model is None:
                reply = "已记下。当前未配置聊天模型。"
            else:
                shortcut = self._faq_shortcut_reply(model, text, history, prep.to_dict())
                if shortcut is not None:
                    reply, meta = shortcut
                else:
                    text, history, observe, _tools, meta, tasks = self._attach_intent(
                        session, text, history, prep, model
                    )
                    hitl = self._maybe_suspend(session, text, meta, skip_ask)
                    if hitl:
                        reply = str(hitl.get("question") or "")
                    else:
                        base_turn = self._turn_payload(
                            session, meta, text, hitl_state=hitl_state, think=think
                        )
                        reply, facts = await _ainvoke_intent_tasks(
                            model, tasks, history, base_turn
                        )
                    meta["langgraph"] = dict(_last_graph_trace.get() or {})
                    tool_row = _pop_tool_hitl_row()
                    if tool_row:
                        hitl = _hitl_response_from_row(tool_row)
                        reply = str((hitl or {}).get("question") or reply)
        finally:
            self._absorb_parsed_report(session)
            companion_user_id.reset(gid)
            companion_last_facts.reset(facts_tok)
            companion_parsed_report.reset(parsed_tok)
        return self._finish_turn(session, text, reply, meta, facts, hitl=hitl)

    async def stream_reply(
        self,
        session: CompanionSession,
        user_text: str,
        resume_token: str = "",
        think: bool = False,
        hitl_decision: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.langchain.agent.stream_agent import sse_payload

        yield sse_payload({"event": "thinking", "message": "正在预处理…"})
        from algorithm.reply_actions import reply_actions

        resumed = await self._resume_tool_hitl_if_any(
            session, user_text, resume_token, hitl_decision, think=think
        )
        if resumed is not None:
            hitl = resumed.get("hitl")
            if hitl and hitl.get("resume_token"):
                yield sse_payload(
                    {
                        "event": "suspended",
                        "hitl_kind": hitl.get("hitl_kind"),
                        "question": hitl.get("question"),
                        "resume_token": hitl.get("resume_token"),
                        "action_requests": hitl.get("action_requests") or [],
                    }
                )
            yield sse_payload({"event": "token", "content": str(resumed.get("reply") or "")})
            yield sse_payload({"event": "state", "data": resumed})
            yield sse_payload({"event": "done"})
            return

        from algorithm.stream_debug import bind_stream_debug_emitter, unbind_stream_debug_emitter

        stream_q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def _stream_put(kind: str, payload: Any) -> None:
            try:
                stream_q.put_nowait((kind, payload))
            except asyncio.QueueFull:
                pass

        debug_tok = bind_stream_debug_emitter(lambda row: _stream_put("debug", row))

        text, history, prep, skip_ask, hitl_state = self._begin_turn(
            session, user_text, resume_token=resume_token
        )
        while not stream_q.empty():
            kind, payload = stream_q.get_nowait()
            if kind == "debug" and isinstance(payload, dict):
                yield sse_payload({"event": "debug", **payload})
        gid = companion_user_id.set(session.group_id)
        facts_tok = companion_last_facts.set([])
        parsed_tok = companion_parsed_report.set(None)
        meta = {
            "search_query": "",
            "skipped_long_memory": True,
            "preprocess": prep.to_dict(),
            "bound_tools": [],
        }
        observe = format_preprocess_for_agent(prep)
        reply = "我在，你可以继续说说近况。"
        facts: list[dict[str, Any]] = []
        hitl: dict[str, Any] | None = None
        try:
            model = _chat_model()
            if model is None:
                reply = "已记下。当前未配置聊天模型。"
                yield sse_payload({"event": "token", "content": reply})
            else:
                shortcut = self._faq_shortcut_reply(model, text, history, prep.to_dict())
                if shortcut is not None:
                    reply, meta = shortcut
                    yield sse_payload({"event": "thinking", "message": "命中固定话术，直接回复…"})
                    yield sse_payload({"event": "token", "content": reply})
                else:
                    yield sse_payload({"event": "thinking", "message": "正在识别意图…"})
                    text, history, observe, tools, meta, tasks = self._attach_intent(
                        session, text, history, prep, model
                    )
                if shortcut is None:
                    while not stream_q.empty():
                        kind, payload = stream_q.get_nowait()
                        if kind == "debug":
                            yield sse_payload({"event": "debug", **payload})
                    hitl = self._maybe_suspend(session, text, meta, skip_ask)
                    if hitl:
                        reply = str(hitl.get("question") or "")
                        yield sse_payload(
                            {
                                "event": "suspended",
                                "hitl_kind": hitl.get("hitl_kind"),
                                "question": reply,
                                "resume_token": hitl.get("resume_token"),
                                "agent_name": hitl.get("agent_name"),
                                "action_requests": hitl.get("action_requests") or [],
                                "actions": reply_actions(
                                    hitl={
                                        "pending": True,
                                        "resume_token": hitl.get("resume_token"),
                                        "question": hitl.get("question"),
                                        "agent_name": hitl.get("agent_name"),
                                    },
                                    preprocess=meta.get("preprocess")
                                    if isinstance(meta.get("preprocess"), dict)
                                    else {},
                                    last_report=session.last_report,
                                    slot_memory=session.slot_memory,
                                ),
                            }
                        )
                        yield sse_payload({"event": "token", "content": reply})
                    else:
                        base_turn = self._turn_payload(
                            session, meta, text, hitl_state=hitl_state, think=think
                        )
                        if think:
                            yield sse_payload({"event": "thinking", "message": "思考模式：工具返回后审核证据…"})
                        if any(task.get("tools") for task in tasks):
                            yield sse_payload({"event": "thinking", "message": "正在并行调用工具…"})
                        def _emit(token: str) -> None:
                            _stream_put("token", token)

                        run = asyncio.create_task(
                            _ainvoke_intent_tasks(model, tasks, history, base_turn, on_token=_emit)
                        )
                        streamed = False
                        while True:
                            getter = asyncio.create_task(stream_q.get())
                            done_set, _ = await asyncio.wait(
                                {getter, run}, return_when=asyncio.FIRST_COMPLETED
                            )
                            if getter in done_set:
                                kind, payload = getter.result()
                                if kind == "token" and payload:
                                    streamed = True
                                    yield sse_payload({"event": "token", "content": payload})
                                elif kind == "debug" and isinstance(payload, dict):
                                    yield sse_payload({"event": "debug", **payload})
                                continue
                            getter.cancel()
                            break
                        reply, facts = await run
                        meta["langgraph"] = dict(_last_graph_trace.get() or {})
                        tool_row = _pop_tool_hitl_row()
                        if tool_row:
                            hitl = _hitl_response_from_row(tool_row)
                            reply = str((hitl or {}).get("question") or reply)
                            yield sse_payload(
                                {
                                    "event": "suspended",
                                    "hitl_kind": hitl.get("hitl_kind") if hitl else "tool_approval",
                                    "question": reply,
                                    "resume_token": (hitl or {}).get("resume_token"),
                                    "agent_name": (hitl or {}).get("agent_name"),
                                    "action_requests": (hitl or {}).get("action_requests") or [],
                                }
                            )
                        while not stream_q.empty():
                            kind, payload = stream_q.get_nowait()
                            if kind == "token" and payload:
                                streamed = True
                                yield sse_payload({"event": "token", "content": payload})
                            elif kind == "debug" and isinstance(payload, dict):
                                yield sse_payload({"event": "debug", **payload})
                        if not streamed and reply.strip():
                            step = 18
                            for i in range(0, len(reply), step):
                                yield sse_payload({"event": "token", "content": reply[i : i + step]})
        except Exception:
            logger.exception("companion stream failed")
            reply = "我先记下了，网络波动稍后再聊也行。"
            yield sse_payload({"event": "token", "content": reply})
        finally:
            self._absorb_parsed_report(session)
            companion_user_id.reset(gid)
            companion_last_facts.reset(facts_tok)
            companion_parsed_report.reset(parsed_tok)
            try:
                unbind_stream_debug_emitter(debug_tok)
            except Exception:
                pass

        payload = self._finish_turn(session, text, reply, meta, facts, hitl=hitl)
        final = str(payload.get("reply") or "")
        mark = "（本轮消耗 token："
        idx = final.rfind(mark)
        if idx >= 0 and mark not in reply:
            yield sse_payload({"event": "token", "content": "\n\n" + final[idx:]})
        yield sse_payload({"event": "state", "data": payload})
        yield sse_payload({"event": "done"})

    def _begin_turn(
        self, session: CompanionSession, user_text: str, resume_token: str = ""
    ) -> tuple[str, list[Any], Any, bool, dict[str, Any]]:
        from algorithm.companion_hitl import consume_hitl

        raw = (user_text or "").strip()
        if not raw:
            raise ValueError("text 不能为空")
        from algorithm.run_trace import start_turn_trace

        start_turn_trace(session_id=session.session_id, user_id=session.group_id, query=raw)
        reset_llm_trace()
        from algorithm.companion_hitl import HITL_KIND_SLOT

        pending = consume_hitl(
            session.session_id,
            resume_token,
            kinds=(HITL_KIND_SLOT,),
            user_text=raw,
        )
        hitl_state: dict[str, Any] = {}
        skip_ask = False
        if pending:
            skip_ask = True
            state = dict(pending.get("state") or {})
            merged = str(state.get("merged_query") or "").strip()
            if merged:
                raw = merged
            else:
                original = str(state.get("query") or "").strip()
                if original:
                    raw = f"{original}。{raw}"
            state["answer"] = user_text
            hitl_state = state
        prior_users = [
            str(item.get("content") or "").strip()
            for item in session.user
            if str(item.get("role_type") or "") == "user" and str(item.get("content") or "").strip()
        ]
        last_user = prior_users[-1] if prior_users else ""
        prep = run_chat_preprocess(
            raw,
            user_id=session.group_id,
            last_preprocess=session.last_preprocess,
            last_user_text=last_user,
            slot_memory=session.slot_memory,
            prior_user_texts=prior_users,
            defer_intent=True,
        )
        text = prep.resolved_text or prep.corrected_text or raw
        session.append(role_type="user", content=text)
        sync_b2c_short_memory(session.session_id, list(session.user[:-1]))
        touch_b2c_session_redis(session.group_id, session_id=session.session_id)
        history = self._history_messages(session)
        return text, history, prep, skip_ask, hitl_state

    @staticmethod
    def _maybe_suspend(
        session: CompanionSession,
        text: str,
        meta: dict[str, Any],
        skip_ask: bool,
    ) -> dict[str, Any] | None:
        from algorithm.companion_hitl import ask_if_needed, save_hitl
        from algorithm.chat_preprocess import CompanionPreprocess

        if skip_ask:
            return None
        prep_raw = meta.get("preprocess") if isinstance(meta.get("preprocess"), dict) else {}
        prep = CompanionPreprocess(
            raw_text=str(prep_raw.get("raw_text") or text),
            corrected_text=str(prep_raw.get("corrected_text") or text),
            resolved_text=str(prep_raw.get("resolved_text") or text),
            intent=str(prep_raw.get("intent") or ""),
            intent_major=str(prep_raw.get("intent_major") or ""),
            intent_handler=str(prep_raw.get("intent_handler") or ""),
            mentions=prep_raw.get("mentions") if isinstance(prep_raw.get("mentions"), dict) else {},
            alignments=list(prep_raw.get("alignments") or []),
        )
        asked = ask_if_needed(
            prep,
            slot_memory=session.slot_memory,
            has_uploaded_file=bool(
                (session.last_file or {}).get("saved_path")
                or (session.last_report or {}).get("filename")
            ),
        )
        if not asked:
            return None
        question, agent_name = asked
        return save_hitl(
            session.session_id,
            question=question,
            agent_name=agent_name,
            state={
                "query": text,
                "major": prep.intent_major,
                "handler": prep.intent_handler,
                "preprocess": prep_raw,
            },
        )

    def _faq_shortcut_reply(
        self,
        model: Any,
        text: str,
        history: list[Any],
        preprocess: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """整句命中固定话术时直接组句，不进意图和主图。"""
        from algorithm.faq_shortcut import match_faq_exact
        from algorithm.llm_trace import record_llm_call, serialize_langchain_messages
        from langchain_core.messages import SystemMessage
        from prompt import load_prompt

        kind = match_faq_exact(text)
        if not kind or model is None:
            return None
        msgs: list[Any] = [SystemMessage(content=load_prompt("companion_faq"))]
        msgs.extend(list(history or [])[-12:])
        msgs.append(HumanMessage(content=text))
        resp = model.invoke(msgs)
        reply = str(getattr(resp, "content", "") or "").strip() or "我在，你可以直接说说近况。"
        record_llm_call(
            "FAQ短路",
            serialize_langchain_messages(msgs),
            response=reply,
            label=kind,
        )
        return reply, {
            "search_query": "",
            "skipped_long_memory": True,
            "preprocess": dict(preprocess or {}),
            "bound_tools": [],
            "faq_shortcut": kind,
            "intent_tasks": [],
        }

    def _attach_intent(
        self,
        session: CompanionSession,
        text: str,
        history: list[Any],
        prep: Any,
        model: Any,
    ) -> tuple[str, list[Any], str, list[BaseTool], dict[str, Any], list[dict[str, Any]]]:
        dialogue = "\n".join(
            f"{'用户' if isinstance(msg, HumanMessage) else '助手'}：{msg.content}"
            for msg in history[-12:]
            if getattr(msg, "content", None)
        )
        turn = attach_intent_to_preprocess(
            prep,
            model,
            ctx=CompanionTurnContext(
                recent_dialogue=dialogue,
                last_file=session.last_file if isinstance(session.last_file, dict) else None,
            ),
            backend="llm",
            apply_file_route=True,
        )
        if not turn.get("ok"):
            items = []
            prep = finish_preprocess_after_intent(prep, items)
        else:
            from algorithm.chat_preprocess import prep_from_dict

            prep = prep_from_dict(turn.get("preprocess") or {}, fallback=prep)
            items = list(turn.get("items") or [])
        session.last_preprocess = prep.to_dict()
        session.merge_slot_memory(session.last_preprocess)
        session.bind_user_extract(session.last_preprocess)
        save_b2c_slot_memory(
            session.group_id,
            session.slot_memory,
            session_id=session.session_id,
        )
        touch_b2c_session_redis(session.group_id, session_id=session.session_id)
        observe = format_preprocess_for_agent(prep)
        file_row = session.last_file if isinstance(session.last_file, dict) else {}
        if file_row.get("mineru_pdf_path"):
            observe += (
                f"\n报告指标解读已备 PDF：{file_row.get('filename') or ''}（{file_row.get('mineru_pdf_path')}）。"
                "explain_uploaded_report 直接 MinerU 拆该 PDF，不要再做版面检测。"
            )
        elif file_row.get("saved_path"):
            observe += (
                f"\n本轮已切块：{file_row.get('filename') or ''}（标题/正文/表在 last_file）。"
                "explain_uploaded_report 只读这些分区抽指标，不要再解析原件。"
            )
        tasks: list[dict[str, Any]] = []
        bound: list[str] = []
        for item in items:
            tools = tools_for_intent(major=item.major, handler=item.handler)
            bound.extend(str(getattr(tool, "name", "") or "") for tool in tools)
            tasks.append(
                {
                    "major": item.major,
                    "intent": item.intent,
                    "handler": item.handler,
                    "tools": tools,
                    "observe": observe,
                }
            )
        meta = {
            "search_query": "",
            "skipped_long_memory": prep.skip_long_memory,
            "preprocess": prep.to_dict(),
            "bound_tools": bound,
            "intent_tasks": [intent_result_to_dict(item) for item in items],
            "recent_dialogue": dialogue,
            "observe": observe,
        }
        tools = list(tasks[0]["tools"]) if tasks else []
        return text, history, observe, tools, meta, tasks

    @staticmethod
    def _absorb_parsed_report(session: CompanionSession) -> None:
        row = companion_parsed_report.get()
        if isinstance(row, dict) and (row.get("filename") or row.get("indicators")):
            session.last_report = row

    @staticmethod
    def _turn_payload(
        session: CompanionSession,
        meta: dict[str, Any],
        text: str,
        hitl_state: dict[str, Any] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        prep = meta.get("preprocess") if isinstance(meta.get("preprocess"), dict) else {}
        from algorithm.run_trace import current_trace_id

        return {
            "user_id": session.group_id,
            "session_id": session.session_id,
            "trace_id": current_trace_id(),
            "handler": prep.get("intent_handler") or "",
            "query": text,
            "mentions": prep.get("mentions") or {},
            "alignments": prep.get("alignments") or [],
            "hitl": dict(hitl_state or {}),
            "last_report": dict(session.last_report or {}),
            "last_file": dict(session.last_file or {}),
            "think": bool(think),
        }

    @staticmethod
    def _history_messages(session: CompanionSession) -> list[Any]:
        sid = str(session.session_id or "").strip()
        if sid:
            packed = load_b2c_agent_history(sid)
            if packed:
                return packed
        history: list[Any] = []
        for item in session.user[:-1][-40:]:
            role = str(item.get("role_type") or item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role in ("human", "user"):
                history.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                history.append(AIMessage(content=content))
        if sid and history:
            from algorithm.context_compress import apply_incremental_compress

            return apply_incremental_compress(history, session_id=sid)
        return history

    @staticmethod
    def _enrich_meta_tool_bindings(meta: dict[str, Any]) -> None:
        from tools.mcp_bridge import split_bound_tools

        graph = meta.get("langgraph") if isinstance(meta.get("langgraph"), dict) else {}
        names: list[str] = list(meta.get("bound_tools") or [])
        for run in graph.get("tool_runs") or []:
            if not isinstance(run, dict):
                continue
            names.extend(str(n) for n in (run.get("bound_tools") or []) if n)
        dedup: list[str] = []
        seen: set[str] = set()
        for name in names:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            dedup.append(n)
        local, mcp = split_bound_tools(dedup)
        meta["bound_tools"] = dedup
        meta["bound_local_tools"] = local
        meta["bound_mcp_tools"] = mcp

    def _finish_turn(
        self,
        session: CompanionSession,
        text: str,
        reply: str,
        meta: dict[str, Any],
        facts: list[dict[str, Any]] | None = None,
        hitl: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from algorithm.llm_trace import turn_token_footer
        from algorithm.reply_actions import reply_actions

        reply = turn_token_footer(reply)
        self._enrich_meta_tool_bindings(meta)
        graph = meta.get("langgraph") if isinstance(meta.get("langgraph"), dict) else {}
        from algorithm.run_trace import build_round_observability, current_trace_id

        observability = build_round_observability(graph)
        if not observability.get("trace_id"):
            observability["trace_id"] = current_trace_id()
        meta["observability"] = observability
        from algorithm.run_trace import log_trace_event

        log_trace_event(
            "turn_end",
            session_id=session.session_id,
            tool_steps=sum(len(r.get("steps") or []) for r in observability.get("tool_runs") or []),
        )
        tid = str(graph.get("thread_id") or session.session_id or "")
        from graph.checkpoint_debug import build_round_checkpoint_debug

        meta["checkpoint_debug"] = build_round_checkpoint_debug(
            session_id=session.session_id,
            companion_thread_id=tid,
            companion_snapshot=graph.get("companion_checkpoint_snapshot"),
        )
        session.append(role_type="assistant", content=reply)
        sync_b2c_session_memory(session)
        facts = list(facts or [])
        ingest = {
            "deferred": True,
            "reason": "persist_on_session_end",
            "persisted": session.persisted,
        }
        session_dump = session.to_dict()
        hitl_out = None
        if hitl:
            hitl_out = {
                "pending": True,
                "hitl_kind": hitl.get("hitl_kind"),
                "resume_token": hitl.get("resume_token"),
                "question": hitl.get("question"),
                "agent_name": hitl.get("agent_name"),
                "action_requests": list(hitl.get("action_requests") or []),
            }
        actions = reply_actions(
            hitl=hitl_out,
            preprocess=meta.get("preprocess") if isinstance(meta.get("preprocess"), dict) else {},
            last_report=session.last_report,
            slot_memory=session.slot_memory,
        )
        return {
            "reply": reply,
            "facts": facts,
            "ingest": ingest,
            "trace_id": observability.get("trace_id") or "",
            "observability": observability,
            "preprocess": meta.get("preprocess"),
            "session": session_dump,
            "hitl": hitl_out,
            "actions": actions,
            "round": {
                "session_id": session.session_id,
                "group_id": session.group_id,
                "user_text": text,
                "reply": reply,
                "memory_facts": facts,
                "search_query": meta.get("search_query"),
                "skipped_long_memory": meta.get("skipped_long_memory"),
                "bound_tools": meta.get("bound_tools") or [],
                "bound_local_tools": meta.get("bound_local_tools") or [],
                "bound_mcp_tools": meta.get("bound_mcp_tools") or [],
                "intent_tasks": list(meta.get("intent_tasks") or []),
                "preprocess": meta.get("preprocess"),
                "langchain_agent": "langgraph+supervisor+review",
                "langgraph": meta.get("langgraph") or {},
                "checkpoint_debug": meta.get("checkpoint_debug") or {},
                "llm_calls": _llm_call_briefs(),
                "rag_enhance": _rag_enhance_briefs(graph),
                "llm_trace": take_llm_trace(),
                "ingest": ingest,
                "trace_id": observability.get("trace_id") or "",
                "observability": observability,
                "hitl": hitl_out,
                "actions": actions,
                "short_memory_turns": len(load_b2c_short_messages(session.session_id) or session.user),
                "session": session_dump,
            },
        }


def _rag_enhance_briefs(graph: dict[str, Any]) -> list[dict[str, str]]:
    from algorithm.kb_eval_flow import format_companion_rag_block

    rows: list[dict[str, str]] = []
    for run in list(graph.get("tool_runs") or []):
        if not isinstance(run, dict):
            continue
        for step in list(run.get("steps") or []):
            if not isinstance(step, dict):
                continue
            env = step.get("envelope") if isinstance(step.get("envelope"), dict) else {}
            data = env.get("data") if isinstance(env.get("data"), dict) else {}
            rag = data.get("rag") if isinstance(data.get("rag"), dict) else None
            source = str(data.get("source") or "")
            tool = str(env.get("tool") or step.get("name") or "")
            if tool not in {"search_exam_handbook", "search_chroma_knowledge", "query_consult_graph"}:
                if not rag:
                    continue
            elif not rag and source not in {"ragflow", "chroma", "neo4j"}:
                continue
            rows.append({"tool": tool or source or "rag", "body": format_companion_rag_block(data, tool=tool)})
    return rows


def _llm_call_briefs() -> list[dict[str, Any]]:
    """调试台：本轮每次 LLM invoke 的完整 messages（见 algorithm.llm_trace）。"""
    return llm_trace_to_briefs()
