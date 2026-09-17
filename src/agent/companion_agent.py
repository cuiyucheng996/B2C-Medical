"""B2C 陪聊：多意图可并行 create_agent，再拼回复。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
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
from algorithm.intent_classify import (
    classify_companion_intent_llm,
    classify_intent_fallback,
    intent_result_to_dict,
    normalize_intent_results,
)
from algorithm.reply_stream import ReplyTextExtractor, tool_call_name
from algorithm.short_memory import load_b2c_short_messages, save_b2c_slot_memory, sync_b2c_short_memory
from middleware import companion_middleware
from middleware.context import CompanionRunContext
from middleware.tool_pairs import drop_orphan_tool_messages
from schema.companion_reply import CompanionReply
from session import CompanionSession
from tools.companion_memory import companion_last_facts, companion_parsed_report, companion_user_id
from tools.companion_tools import companion_turn, tools_for_intent
from tools.mcp_bridge import mcp_tools_for_major

logger = logging.getLogger(__name__)

__all__ = ["CompanionAgent", "create_companion_agent"]


def create_companion_agent(model: Any, tools: list[BaseTool] | None = None) -> Any:
    """本轮工具集不同；提示词、观察、MCP 护栏、信封、重试均走中间件。

    有业务工具时不要把 CompanionReply 再做成一把 tool（ToolStrategy），
    兼容口会和真工具抢 role=tool，整包 400。无工具时才用 ToolStrategy 组句。
    """
    bound = list(tools or [])
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": bound,
        "middleware": companion_middleware(),
        "context_schema": CompanionRunContext,
        "name": "companion_chat",
    }
    if not bound:
        kwargs["response_format"] = ToolStrategy(CompanionReply)
    return create_agent(**kwargs)


def _chat_model() -> Any:
    from configs.paths_b2c import ensure_uvgraph_root

    ensure_uvgraph_root()
    from src.llm.chat import get_chat_model

    return get_chat_model()


def _reply_from_agent_result(result: dict[str, Any], *, streamed: str = "") -> str:
    structured = result.get("structured_response")
    if isinstance(structured, CompanionReply):
        text = (structured.assistant_reply or "").strip()
        if text:
            return text
    if isinstance(structured, dict):
        text = str(structured.get("assistant_reply") or "").strip()
        if text:
            return text
    if streamed.strip():
        # 结构化结果没拿到时，用已经流出去的那段，避免前端看到的内容被改写
        return streamed.strip()
    for msg in reversed(list(result.get("messages") or [])):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = str(getattr(msg, "content", "") or "").strip()
        if text:
            return text
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
) -> tuple[str, list[dict[str, Any]]]:
    turn_tok = None
    facts_tok = companion_last_facts.set([])
    if turn:
        turn_tok = companion_turn.set(turn)
    result: Any = None
    streamed = ""
    row = task or {}
    ctx = CompanionRunContext(
        query=str((turn or {}).get("query") or ""),
        observe=observe,
        major=str(row.get("major") or ""),
        intent=str(row.get("intent") or ""),
        handler=str(row.get("handler") or (turn or {}).get("handler") or ""),
        user_id=str((turn or {}).get("user_id") or ""),
        session_id=str((turn or {}).get("session_id") or ""),
        think_enabled=bool((turn or {}).get("think")),
    )
    human = ctx.query or "请根据预处理观察调用工具并回复。"
    try:
        agent = create_companion_agent(model, tools)
        payload = {
            "messages": drop_orphan_tool_messages([*history, HumanMessage(content=human)])
        }
        if on_token is None:
            result = await agent.ainvoke(payload, context=ctx)
        else:
            result, streamed = await _astream_companion_agent(agent, payload, ctx, on_token)
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
                        fallback, packed, ctx, on_token
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
        return "我先记下了，网络波动稍后再聊也行。", facts
    if not isinstance(result, dict):
        return "我在，你可以继续说说近况。", facts
    return _reply_from_agent_result(result, streamed=streamed), facts


async def _astream_companion_agent(
    agent: Any,
    payload: dict[str, Any],
    ctx: CompanionRunContext,
    on_token: Any,
) -> tuple[Any, str]:
    """流式跑 agent：只把 CompanionReply 的 assistant_reply 增量吐给 on_token。

    结构化输出走 tool calling，直接透传 token 会露出 JSON，所以边收边解码。
    真正的工具调用（note_task_* 等）参数不吐。
    """
    extractor = ReplyTextExtractor()
    result: Any = None
    content_bits: list[str] = []
    async for event in agent.astream_events(payload, context=ctx, version="v2"):
        kind = event.get("event")
        if kind == "on_chat_model_start":
            # 新一轮模型调用，重新累积（工具调用后的最终回复才是给用户看的）
            extractor.reset()
            content_bits.clear()
            continue
        if kind != "on_chat_model_stream":
            if kind == "on_chain_end" and event.get("name") == "companion_chat":
                data = (event.get("data") or {}).get("output")
                if isinstance(data, dict):
                    result = data
            continue
        chunk = (event.get("data") or {}).get("chunk")
        name = tool_call_name(chunk)
        if name not in ("", "CompanionReply"):
            continue
        text = getattr(chunk, "content", None)
        if name == "" and isinstance(text, str) and text:
            on_token(text)
            content_bits.append(text)
            continue
        for item in getattr(chunk, "tool_call_chunks", None) or []:
            delta = extractor.feed_args(str((item or {}).get("args") or ""))
            if delta:
                on_token(delta)
    return result, extractor.text or "".join(content_bits)


async def _ainvoke_intent_tasks(
    model: Any,
    tasks: list[dict[str, Any]],
    history: list[Any],
    base_turn: dict[str, Any],
    on_token: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    if not tasks:
        return "我在，你可以继续说说近况。", []

    async def _one(task: dict[str, Any], emit: Any = None) -> tuple[str, list[dict[str, Any]]]:
        turn = {
            **base_turn,
            "handler": task.get("handler") or base_turn.get("handler") or "",
        }
        observe = str(task.get("observe") or "")
        local = list(task.get("tools") or [])
        extra = await mcp_tools_for_major(str(task.get("major") or ""))
        have = {str(getattr(tool, "name", "") or "") for tool in local}
        tools = local + [tool for tool in extra if str(getattr(tool, "name", "") or "") not in have]
        return await _ainvoke_companion_agent(
            model, tools, history, observe, turn, task, on_token=emit
        )

    if len(tasks) == 1:
        return await _one(tasks[0], on_token)
    # 多意图并行：几个模型同时吐字会互相串行，这里不流式，只拼最终结果
    gathered = await asyncio.gather(*[_one(task) for task in tasks], return_exceptions=True)
    replies: list[str] = []
    facts: list[dict[str, Any]] = []
    for item in gathered:
        if isinstance(item, BaseException):
            logger.exception("companion intent task failed", exc_info=item)
            continue
        replies.append(item[0])
        facts.extend(item[1])
    text = "\n\n".join(part.strip() for part in replies if str(part).strip())
    return text or "我在，你可以继续说说近况。", facts


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("sync companion reply cannot run inside a running event loop")


class CompanionAgent:
    def reply(
        self,
        session: CompanionSession,
        user_text: str,
        resume_token: str = "",
        think: bool = False,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """已有事件循环里用：不要走 reply() 的 asyncio.run。"""
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
    ) -> AsyncIterator[str]:
        from configs.paths_b2c import ensure_uvgraph_root

        ensure_uvgraph_root()
        from src.langchain.agent.stream_agent import sse_payload

        yield sse_payload({"event": "thinking", "message": "正在预处理…"})
        from algorithm.reply_actions import reply_actions

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
        reply = "我在，你可以继续说说近况。"
        facts: list[dict[str, Any]] = []
        hitl: dict[str, Any] | None = None
        try:
            model = _chat_model()
            if model is None:
                reply = "已记下。当前未配置聊天模型。"
                yield sse_payload({"event": "token", "content": reply})
            else:
                yield sse_payload({"event": "thinking", "message": "正在识别意图…"})
                text, history, observe, tools, meta, tasks = self._attach_intent(
                    session, text, history, prep, model
                )
                hitl = self._maybe_suspend(session, text, meta, skip_ask)
                if hitl:
                    reply = str(hitl.get("question") or "")
                    yield sse_payload(
                        {
                            "event": "suspended",
                            "question": reply,
                            "resume_token": hitl.get("resume_token"),
                            "agent_name": hitl.get("agent_name"),
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
                    queue: asyncio.Queue[str] = asyncio.Queue()

                    def _emit(token: str) -> None:
                        queue.put_nowait(token)

                    run = asyncio.create_task(
                        _ainvoke_intent_tasks(model, tasks, history, base_turn, on_token=_emit)
                    )
                    # 边跑边把 token 推给前端；run 结束时排空队列再收尾
                    streamed = False
                    while True:
                        getter = asyncio.create_task(queue.get())
                        done_set, _ = await asyncio.wait(
                            {getter, run}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if getter in done_set:
                            token = getter.result()
                            if token:
                                streamed = True
                                yield sse_payload({"event": "token", "content": token})
                            continue
                        getter.cancel()
                        break
                    reply, facts = await run
                    while not queue.empty():
                        token = queue.get_nowait()
                        if token:
                            streamed = True
                            yield sse_payload({"event": "token", "content": token})
                    if not streamed and reply.strip():
                        # 多意图并行或模型没走 CompanionReply，整段补发一次
                        yield sse_payload({"event": "token", "content": reply})
        except Exception:
            logger.exception("companion stream failed")
            reply = "我先记下了，网络波动稍后再聊也行。"
            yield sse_payload({"event": "token", "content": reply})
        finally:
            self._absorb_parsed_report(session)
            companion_user_id.reset(gid)
            companion_last_facts.reset(facts_tok)
            companion_parsed_report.reset(parsed_tok)

        payload = self._finish_turn(session, text, reply, meta, facts, hitl=hitl)
        yield sse_payload({"event": "state", "data": payload})
        yield sse_payload({"event": "done"})

    def _begin_turn(
        self, session: CompanionSession, user_text: str, resume_token: str = ""
    ) -> tuple[str, list[Any], Any, bool, dict[str, Any]]:
        from algorithm.companion_hitl import consume_hitl

        raw = (user_text or "").strip()
        if not raw:
            raise ValueError("text 不能为空")
        pending = consume_hitl(session.group_id, resume_token)
        hitl_state: dict[str, Any] = {}
        skip_ask = False
        if pending:
            skip_ask = True
            state = dict(pending.get("state") or {})
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
        sync_b2c_short_memory(session.group_id, list(session.user[:-1]))
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
            session.group_id,
            question=question,
            agent_name=agent_name,
            state={
                "query": text,
                "major": prep.intent_major,
                "handler": prep.intent_handler,
                "preprocess": prep_raw,
            },
        )

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
        classified = classify_companion_intent_llm(model, text, recent_dialogue=dialogue)
        if classified is None:
            classified = [classify_intent_fallback(text)]
        items = normalize_intent_results(list(classified))
        items = self._apply_file_route(session, text, items)
        prep = finish_preprocess_after_intent(prep, items)
        session.last_preprocess = prep.to_dict()
        session.merge_slot_memory(session.last_preprocess)
        session.bind_user_extract(session.last_preprocess)
        save_b2c_slot_memory(
            session.group_id,
            session.slot_memory,
            session_id=session.session_id,
        )
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
        }
        tools = list(tasks[0]["tools"]) if tasks else []
        return text, history, observe, tools, meta, tasks

    @staticmethod
    def _absorb_parsed_report(session: CompanionSession) -> None:
        row = companion_parsed_report.get()
        if isinstance(row, dict) and (row.get("filename") or row.get("indicators")):
            session.last_report = row

    @staticmethod
    def _apply_file_route(
        session: CompanionSession, text: str, items: list[Any]
    ) -> list[Any]:
        """上传原件：打分只锁大类。化验单细意图交给 LLM；其它仍走图片/文档细类。"""
        from algorithm.file_gate import intent_for_file_role
        from algorithm.intent_classify import CompanionIntentResult, normalize_intent_results

        prefix = (text or "").lstrip()
        uploadish = prefix.startswith(("【上传检查报告】", "【上传图片】", "【上传文档】"))
        row = session.last_file if isinstance(session.last_file, dict) else {}
        if not uploadish and not row.get("saved_path") and str(row.get("role") or "") != "layout_picture":
            return items
        spec = intent_for_file_role(str(row.get("role") or ""))
        if not spec:
            return items
        if spec["handler"] == "report_upload":
            kept = [item for item in items if getattr(item, "major", "") == "报告指标解读"]
            if kept:
                return normalize_intent_results(kept)
            forced = CompanionIntentResult(
                intent="未细分",
                major="报告指标解读",
                confidence=1.0,
                source="file_gate",
                reason="；".join(list(row.get("reasons") or [])[:4]) or "文件前置分流",
                slots=(),
                handler="fallback",
            )
            return normalize_intent_results([forced])
        forced = CompanionIntentResult(
            intent=spec["intent"],
            major=spec["major"],
            confidence=1.0,
            source="file_gate",
            reason="；".join(list(row.get("reasons") or [])[:4]) or "文件前置分流",
            slots=(),
            handler=spec["handler"],
        )
        return normalize_intent_results([forced])

    @staticmethod
    def _turn_payload(
        session: CompanionSession,
        meta: dict[str, Any],
        text: str,
        hitl_state: dict[str, Any] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        prep = meta.get("preprocess") if isinstance(meta.get("preprocess"), dict) else {}
        return {
            "user_id": session.group_id,
            "session_id": session.session_id,
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
        rows = load_b2c_short_messages(session.group_id)
        if not rows:
            rows = []
            for item in session.user[:-1][-40:]:
                role = str(item.get("role_type") or item.get("role") or "")
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                if role in ("human", "user"):
                    rows.append({"role": "user", "content": content})
                elif role in ("ai", "assistant"):
                    rows.append({"role": "assistant", "content": content})
        history: list[Any] = []
        for item in rows:
            content = str(item.get("content") or "")
            if not content:
                continue
            if item.get("role") == "user":
                history.append(HumanMessage(content=content))
            elif item.get("role") == "assistant":
                history.append(AIMessage(content=content))
        return history

    def _finish_turn(
        self,
        session: CompanionSession,
        text: str,
        reply: str,
        meta: dict[str, Any],
        facts: list[dict[str, Any]] | None = None,
        hitl: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from algorithm.reply_actions import reply_actions

        session.append(role_type="assistant", content=reply)
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
                "resume_token": hitl.get("resume_token"),
                "question": hitl.get("question"),
                "agent_name": hitl.get("agent_name"),
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
                "preprocess": meta.get("preprocess"),
                "langchain_agent": "create_agent+async_intents",
                "ingest": ingest,
                "hitl": hitl_out,
                "actions": actions,
                "short_memory_turns": len(load_b2c_short_messages(session.group_id)),
                "session": session_dump,
            },
        }
