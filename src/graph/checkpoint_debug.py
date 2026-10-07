"""陪聊调试：checkpointer 实例与 thread 快照（仅展示，不写库）。"""

from __future__ import annotations

import json
import os
from typing import Any

from configs.paths_b2c import ensure_uvgraph_root

__all__ = [
    "CheckpointVersionStreamer",
    "ainvoke_graph_emitting_checkpoints",
    "build_round_checkpoint_debug",
    "list_checkpoint_versions",
    "snapshot_companion_checkpoint_debug",
    "snapshot_hitl_checkpoint_debug",
]


def _json_preview(value: Any, *, limit: int = 2800) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "\n…（已截断）"
    return text


def _describe_saver(saver: Any, *, env_hint: str) -> dict[str, Any]:
    if saver is None:
        return {"enabled": False, "detail": "实例为 None（未挂载 checkpointer）"}
    cls = type(saver).__name__
    mode = (os.getenv(env_hint) or "").strip() or "（默认 memory）"
    row: dict[str, Any] = {
        "enabled": True,
        "class": cls,
        "config_env": f"{env_hint}={mode}",
    }
    storage = getattr(saver, "storage", None)
    if isinstance(storage, dict):
        row["thread_count"] = len(storage)
        row["thread_ids"] = sorted(storage.keys())[:24]
    elif storage is not None and hasattr(storage, "keys"):
        try:
            keys = list(storage.keys())[:24]
            row["thread_ids"] = keys
            row["thread_count"] = len(keys)
        except Exception:
            pass
    return row


def _message_snapshot(msg: Any, *, limit: int = 1000) -> Any:
    if msg is None:
        return None
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        role = msg.get("type") or msg.get("role")
        content = msg.get("content")
    if isinstance(content, list):
        try:
            content = json.dumps(content, ensure_ascii=False, default=str)
        except TypeError:
            content = str(content)
    text = str(content or "")
    if len(text) > limit:
        text = text[:limit] + "…"
    row: dict[str, Any] = {}
    if role:
        row["role"] = str(role)
    if text:
        row["content"] = text
    name = getattr(msg, "name", None) if not isinstance(msg, dict) else msg.get("name")
    if name:
        row["name"] = str(name)
    tool_calls = getattr(msg, "tool_calls", None) if not isinstance(msg, dict) else msg.get("tool_calls")
    if tool_calls:
        try:
            row["tool_calls"] = json.loads(json.dumps(tool_calls, ensure_ascii=False, default=str))
        except TypeError:
            row["tool_calls"] = str(tool_calls)[:limit]
    if isinstance(msg, dict) and not row:
        return {k: _json_preview(v, limit=limit) for k, v in list(msg.items())[:24]}
    return row or str(msg)[:limit]


def _snapshot_value(key: str, val: Any, *, list_item_limit: int = 12) -> Any:
    if val is None:
        return None
    if key == "history" and isinstance(val, list):
        return [_message_snapshot(item) for item in val[:list_item_limit]] + (
            [f"…共 {len(val)} 条，仅展示前 {list_item_limit} 条"] if len(val) > list_item_limit else []
        )
    if key in {"worker_outputs", "facts", "tasks", "wave1_outputs"} and isinstance(val, list):
        try:
            packed = json.loads(json.dumps(val[:list_item_limit], ensure_ascii=False, default=str))
        except TypeError:
            packed = [str(item)[:800] for item in val[:list_item_limit]]
        if len(val) > list_item_limit:
            packed.append({"_truncated": f"共 {len(val)} 条"})
        return packed
    if key == "messages" and isinstance(val, list):
        return [_message_snapshot(item) for item in val[:list_item_limit]]
    if isinstance(val, (str, int, float, bool)):
        s = str(val)
        return s[:2400] + ("…" if len(s) > 2400 else "")
    if isinstance(val, dict):
        try:
            return json.loads(json.dumps(val, ensure_ascii=False, default=str))
        except TypeError:
            return _json_preview(val, limit=2400)
    if isinstance(val, list):
        try:
            return json.loads(json.dumps(val[:list_item_limit], ensure_ascii=False, default=str))
        except TypeError:
            return [str(item)[:400] for item in val[:list_item_limit]]
    return _json_preview(val, limit=1200)


def _channel_values_snapshot(values: dict[str, Any] | None) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for key, val in dict(values or {}).items():
        snap[key] = _snapshot_value(str(key), val)
    return snap


def _preview_channel_values(values: dict[str, Any] | None) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, val in dict(values or {}).items():
        if key in {"history", "worker_outputs", "facts", "messages"}:
            if isinstance(val, list):
                preview[key] = f"list(len={len(val)})"
            else:
                preview[key] = type(val).__name__
        elif key == "trace" and isinstance(val, dict):
            preview[key] = {k: val.get(k) for k in ("engine", "workers", "pattern", "thread_id") if k in val}
        elif isinstance(val, dict):
            preview[key] = f"dict(keys={list(val.keys())[:12]})"
        else:
            s = str(val)
            preview[key] = s[:200] + ("…" if len(s) > 200 else "")
    return preview


def _task_paths_from_writes(
    saver: Any,
    *,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
) -> list[str]:
    writes_map = getattr(saver, "writes", None)
    if not isinstance(writes_map, dict):
        return []
    bucket = writes_map.get((thread_id, checkpoint_ns, checkpoint_id))
    if not bucket:
        return []
    paths: list[str] = []
    for row in bucket.values():
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            continue
        path = str(row[3] or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _node_hint(
    *,
    metadata: dict[str, Any],
    channel_values: dict[str, Any],
    task_paths: list[str],
) -> str:
    if task_paths:
        # task_path 形如 "execute,review" 或 Pregel 路径，取最后一段作节点名
        labels: list[str] = []
        for path in task_paths[:4]:
            tail = path.replace("|", ",").split(",")[-1].strip()
            if tail and tail not in labels:
                labels.append(tail)
        if labels:
            return "节点 " + " / ".join(labels)
    versions_seen = metadata.get("versions_seen") if isinstance(metadata.get("versions_seen"), dict) else {}
    if isinstance(channel_values, dict) and channel_values:
        keys = [k for k in channel_values if k not in {"messages"}][:6]
        if keys:
            return "状态通道 " + "、".join(keys)
    if isinstance(versions_seen, dict) and versions_seen:
        nodes = sorted(versions_seen.keys())[-4:]
        return "图节点 " + "、".join(nodes)
    source = str(metadata.get("source") or "")
    step = metadata.get("step")
    if source:
        return f"{source} step={step}"
    return "（节点未知）"


def _version_row_from_tuple(
    tup: Any,
    checkpointer: Any,
    *,
    thread_id: str,
) -> dict[str, Any]:
    cfg = tup.config if hasattr(tup, "config") else {}
    conf = cfg.get("configurable") if isinstance(cfg, dict) else {}
    cp_id = str(conf.get("checkpoint_id") or "")
    meta = dict(tup.metadata or {}) if hasattr(tup, "metadata") else {}
    cp_body = dict(tup.checkpoint or {}) if hasattr(tup, "checkpoint") else {}
    if isinstance(cp_body.get("versions_seen"), dict):
        meta = {**meta, "versions_seen": cp_body.get("versions_seen")}
    ns = str(conf.get("checkpoint_ns") or "")
    paths = _task_paths_from_writes(
        checkpointer,
        thread_id=thread_id,
        checkpoint_ns=ns,
        checkpoint_id=cp_id,
    )
    channel_values = cp_body.get("channel_values")
    if not isinstance(channel_values, dict):
        channel_values = {}
    pending = list(tup.pending_writes or []) if getattr(tup, "pending_writes", None) else []
    pending_channels = [str(item[1]) for item in pending if isinstance(item, (list, tuple)) and len(item) > 1]
    pending_snapshot: list[Any] = []
    for item in pending[:16]:
        if not isinstance(item, (list, tuple)):
            pending_snapshot.append(str(item)[:400])
            continue
        try:
            pending_snapshot.append(json.loads(json.dumps(item, ensure_ascii=False, default=str)))
        except TypeError:
            pending_snapshot.append([str(x)[:400] for x in item])
    meta_snap: dict[str, Any] = {}
    if meta:
        try:
            meta_snap = json.loads(json.dumps(meta, ensure_ascii=False, default=str))
        except TypeError:
            meta_snap = {"raw": str(meta)[:800]}
    return {
        "checkpoint_id": cp_id,
        "checkpoint_ns": ns,
        "ts": str(cp_body.get("ts") or ""),
        "source": meta.get("source"),
        "step": meta.get("step"),
        "node_hint": _node_hint(metadata=meta, channel_values=channel_values, task_paths=paths),
        "task_paths": paths[:6],
        "pending_write_channels": pending_channels[:12],
        "pending_writes_snapshot": pending_snapshot,
        "metadata_snapshot": meta_snap,
        "channel_keys": list(channel_values.keys())[:20],
        "values_preview": _preview_channel_values(channel_values),
        "channel_snapshot": _channel_values_snapshot(channel_values),
    }


def _versions_from_storage(
    checkpointer: Any,
    thread_id: str,
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """list() 无结果时从 InMemorySaver.storage 回退枚举。"""
    tid = (thread_id or "").strip()
    storage = getattr(checkpointer, "storage", None)
    if not tid or not isinstance(storage, dict) or tid not in storage:
        return []
    get_tuple = getattr(checkpointer, "get_tuple", None)
    if not callable(get_tuple):
        return []
    collected: list[dict[str, Any]] = []
    for checkpoint_ns, ns_map in storage[tid].items():
        if not isinstance(ns_map, dict):
            continue
        for cp_id in sorted(ns_map.keys()):
            if limit and len(collected) >= limit:
                break
            cfg = {
                "configurable": {
                    "thread_id": tid,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cp_id,
                }
            }
            tup = get_tuple(cfg)
            if tup is None:
                continue
            collected.append(_version_row_from_tuple(tup, checkpointer, thread_id=tid))
    return collected


def list_checkpoint_versions(
    checkpointer: Any,
    config: dict[str, Any],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """按 checkpoint_id 升序列出 thread 上每个版本（释放前调用才有数据）。"""
    if checkpointer is None:
        return []
    tid = str((config.get("configurable") or {}).get("thread_id") or "")
    collected: list[dict[str, Any]] = []
    list_fn = getattr(checkpointer, "list", None)
    if callable(list_fn):
        try:
            for tup in list_fn(config, limit=limit):
                collected.append(_version_row_from_tuple(tup, checkpointer, thread_id=tid))
        except Exception as exc:
            return [{"seq": 0, "error": str(exc)[:400]}]
    if not collected and tid:
        collected = _versions_from_storage(checkpointer, tid, limit=limit)
    collected.sort(key=lambda row: str(row.get("checkpoint_id") or ""))
    for idx, row in enumerate(collected, start=1):
        row["seq"] = idx
    return collected


def _state_summary(graph: Any, config: dict[str, Any]) -> dict[str, Any]:
    if graph is None:
        return {"empty": True, "detail": "图未编译"}
    try:
        snap = graph.get_state(config)
    except Exception as exc:
        return {"error": str(exc)[:400]}
    if snap is None:
        return {"empty": True, "detail": "无 checkpoint 快照"}
    values = dict(snap.values or {}) if snap.values else {}
    preview: dict[str, Any] = {}
    for key, val in values.items():
        if key in {"history", "worker_outputs", "facts"}:
            if isinstance(val, list):
                preview[key] = f"list(len={len(val)})"
            else:
                preview[key] = type(val).__name__
        elif key == "trace" and isinstance(val, dict):
            preview[key] = {k: val.get(k) for k in ("engine", "workers", "pattern", "thread_id") if k in val}
        else:
            s = str(val)
            preview[key] = s[:240] + ("…" if len(s) > 240 else "")
    interrupts: list[str] = []
    for item in list(getattr(snap, "interrupts", None) or []):
        raw = getattr(item, "value", item)
        if isinstance(raw, dict):
            interrupts.append(
                _json_preview(
                    {
                        "hitl_kind": raw.get("hitl_kind"),
                        "question": raw.get("question"),
                        "agent_name": raw.get("agent_name"),
                    },
                    limit=800,
                )
            )
        else:
            interrupts.append(str(raw)[:400])
    return {
        "next": list(snap.next or []),
        "created_at": str(getattr(snap, "created_at", "") or ""),
        "values_preview": preview,
        "values_snapshot": _channel_values_snapshot(values),
        "interrupts": interrupts,
    }


def snapshot_companion_checkpoint_debug(thread_id: str) -> dict[str, Any]:
    ensure_uvgraph_root()
    from graph.checkpointer import (
        companion_checkpoint_enabled,
        companion_invoke_config,
        get_companion_checkpointer,
    )

    tid = (thread_id or "").strip() or "companion:anonymous"
    cfg = companion_invoke_config(tid)
    cp = get_companion_checkpointer()
    block: dict[str, Any] = {
        "checkpointer": _describe_saver(cp, env_hint="COMPANION_CHECKPOINTER"),
        "graph_enabled": companion_checkpoint_enabled(),
        "thread_id": tid,
        "retention": (os.getenv("COMPANION_CHECKPOINT_RETENTION") or "run").strip().lower(),
        "captured_before_release": True,
    }
    if not companion_checkpoint_enabled() or cp is None:
        block["graph_state"] = {"detail": "主图 checkpointer 未启用"}
        block["checkpoint_versions"] = []
        return block
    from graph.companion import build_companion_graph

    graph = build_companion_graph()
    cp_live = getattr(graph, "checkpointer", None) or cp
    block["graph_state"] = _state_summary(graph, cfg)
    block["checkpoint_versions"] = list_checkpoint_versions(cp_live, cfg)
    if not block["checkpoint_versions"]:
        storage = getattr(cp_live, "storage", None)
        if isinstance(storage, dict):
            block["debug_storage_thread_ids"] = sorted(storage.keys())[:40]
        block["versions_note"] = (
            "无版本记录：未进主图、COMPANION_CHECKPOINTER=none、或 retention=run 释放后读取。"
            " 图跑完时应带 companion_checkpoint_snapshot（释放前抓取）。"
        )
    return block


def snapshot_hitl_checkpoint_debug(session_id: str) -> dict[str, Any]:
    sid = (session_id or "").strip()
    from algorithm.hitl_checkpoint import (
        get_pending_hitl,
        get_hitl_checkpointer,
        meta_thread_id,
        slot_thread_id,
    )

    cp = get_hitl_checkpointer()
    block: dict[str, Any] = {
        "checkpointer": _describe_saver(cp, env_hint="COMPANION_HITL_CHECKPOINTER"),
        "session_id": sid or "（无）",
    }
    if not sid:
        block["hitl_status"] = "没有 HITL 触发（无 session_id）"
        return block

    slot_tid = slot_thread_id(sid)
    meta_tid = meta_thread_id(sid)
    block["slot_thread_id"] = slot_tid
    block["meta_thread_id"] = meta_tid
    from algorithm import hitl_checkpoint as hc

    block["slot_graph_state"] = _state_summary(hc._slot_graph(), hc._config(slot_tid))
    block["meta_graph_state"] = _state_summary(hc._meta_graph(), hc._config(meta_tid))
    block["slot_checkpoint_versions"] = list_checkpoint_versions(cp, hc._config(slot_tid))
    block["meta_checkpoint_versions"] = list_checkpoint_versions(cp, hc._config(meta_tid))

    pending = get_pending_hitl(sid)
    if pending:
        block["hitl_status"] = "有待办 HITL"
        block["pending"] = {
            "hitl_kind": pending.get("hitl_kind"),
            "question": pending.get("question"),
            "agent_name": pending.get("agent_name"),
            "resume_token": pending.get("resume_token"),
            "action_requests": pending.get("action_requests"),
        }
    else:
        block["hitl_status"] = "没有 HITL 触发"
        block["pending"] = None
    return block


class CheckpointVersionStreamer:
    """轮询 checkpointer，对新 checkpoint_id 调用 emit_checkpoint_version。"""

    def __init__(self, checkpointer: Any, config: dict[str, Any]) -> None:
        self._cp = checkpointer
        self._config = config
        self._seen: set[str] = set()

    def poll_and_emit(self) -> None:
        if self._cp is None:
            return
        from algorithm.stream_debug import emit_checkpoint_version

        for row in list_checkpoint_versions(self._cp, self._config):
            cid = str(row.get("checkpoint_id") or "")
            ns = str(row.get("checkpoint_ns") or "")
            key = f"{ns}\x1f{cid}" if cid else f"seq-{row.get('seq')}"
            if key in self._seen:
                continue
            self._seen.add(key)
            emit_checkpoint_version(row)


async def ainvoke_graph_emitting_checkpoints(
    graph: Any,
    payload: dict[str, Any],
    invoke_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """astream updates 过程中流式推送 checkpoint 版本；返回图终态 values。"""
    cp = getattr(graph, "checkpointer", None)
    streamer = CheckpointVersionStreamer(cp, invoke_config) if cp and invoke_config else None
    if invoke_config is not None:
        async for _ in graph.astream(payload, config=invoke_config, stream_mode="updates"):
            if streamer:
                streamer.poll_and_emit()
        if streamer:
            streamer.poll_and_emit()
        if hasattr(graph, "aget_state"):
            snap = await graph.aget_state(invoke_config)
        else:
            snap = graph.get_state(invoke_config)
        if snap is not None and getattr(snap, "values", None) is not None:
            return dict(snap.values or {})
        return {}
    out = await graph.ainvoke(payload)
    return dict(out) if isinstance(out, dict) else {}


def build_round_checkpoint_debug(
    *,
    session_id: str,
    companion_thread_id: str,
    companion_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """合并主图 + HITL 两个 checkpointer 调试块。"""
    companion = companion_snapshot if isinstance(companion_snapshot, dict) else None
    if companion is None:
        companion = snapshot_companion_checkpoint_debug(companion_thread_id)
        companion["captured_before_release"] = False
        companion["read_after_release"] = True
    hitl = snapshot_hitl_checkpoint_debug(session_id)
    return {
        "companion_checkpointer": companion,
        "hitl_checkpointer": hitl,
        "note": "get_agent_checkpointer() 与 get_hitl_checkpointer() 为同一 InMemory/Sqlite 实例（工具 HITL 与 slot HITL 共用）。",
    }
