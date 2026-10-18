"""定向采集：通过影刀 MCP runApp 启动工作流（Excel 完成后仍走 webhook/start-crawl）。"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from server.builtin_server.directed_crawl import config as dc_config

logger = logging.getLogger(__name__)


def _flow_for_profile(profile: str) -> dict[str, Any] | None:
    sources = dc_config.load_sources()
    flows = sources.get("rpa_flows")
    if not isinstance(flows, dict):
        return None
    block = flows.get(profile)
    return block if isinstance(block, dict) else None


def start_rpa_flow(
    *,
    profile: str,
    params: dict[str, Any] | None = None,
    session_id: str = "",
    save_job_fn: Any,
) -> dict[str, Any]:
    flow = _flow_for_profile(profile)
    if not flow:
        return {
            "ok": False,
            "error": f"profile「{profile}」未配置 rpa_flows，请在 yingdao/sources.yaml 填写影刀应用",
        }
    app_uuid = str(flow.get("robot_app_uuid") or "").strip()
    if not app_uuid:
        return {
            "ok": False,
            "error": "robot_app_uuid 为空：在 sources.yaml 的 rpa_flows 中填入影刀应用 UUID",
        }
    merged: dict[str, Any] = {}
    defaults = flow.get("default_params")
    if isinstance(defaults, dict):
        merged.update(defaults)
    if params:
        merged.update(params)

    job_id = uuid.uuid4().hex[:16]
    payload: dict[str, Any] = {
        "job_id": job_id,
        "kind": "rpa_yingdao",
        "profile": profile,
        "session_id": session_id,
        "robot_app_uuid": app_uuid,
        "app_params": merged,
        "status": "starting",
        "title": flow.get("title"),
        "webhook_url": flow.get("webhook_url") or "http://127.0.0.1:8014/webhook/start-crawl",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_job_fn(payload)

    def _worker() -> None:
        from server.builtin_server.directed_crawl.yingdao_mcp import invoke_run_app

        t0 = time.perf_counter()
        try:
            asyncio.run(invoke_run_app(app_uuid, merged))
            payload["status"] = "triggered"
            payload["yingdao_message"] = "runApp 已调用；等待影刀流程结束并回调 webhook"
            payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
        except Exception as exc:
            logger.exception("yingdao runApp profile=%s job=%s", profile, job_id)
            payload["status"] = "failed"
            payload["error"] = str(exc)
            payload["elapsed_s"] = round(time.perf_counter() - t0, 2)
        save_job_fn(payload)

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "ok": True,
        "job_id": job_id,
        "profile": profile,
        "robot_app_uuid": app_uuid,
        "app_params": merged,
        "status": "starting",
        "hint": "影刀在本机执行工作流；结束后在流程里 POST webhook（excel_path 须在 rpa_output_dirs）",
        "webhook_url": payload["webhook_url"],
    }
