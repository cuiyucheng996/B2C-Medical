"""stdio MCP：当前时间与时区换算。Windows 上替代 npx @modelcontextprotocol/server-time。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import companion_mcp._bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("time")
_CN = timezone(timedelta(hours=8))


def _tz(name: str):
    raw = (name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(raw)
    except Exception:
        if raw in {"Asia/Shanghai", "Asia/Hong_Kong", "PRC"}:
            return _CN
        return timezone.utc


@mcp.tool()
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """返回指定时区的当前日期时间。"""
    now = datetime.now(_tz(timezone))
    return now.isoformat(timespec="seconds")


@mcp.tool()
def convert_time(time: str, source_timezone: str, target_timezone: str) -> str:
    """把时间从源时区换到目标时区。time 为 ISO 或 HH:MM。"""
    raw = (time or "").strip()
    src = _tz(source_timezone)
    dst = _tz(target_timezone)
    parsed: datetime
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        today = datetime.now(src)
        try:
            hh, mm = raw.split(":", 1)
            parsed = today.replace(hour=int(hh), minute=int(mm.split(":")[0]), second=0, microsecond=0)
        except (ValueError, TypeError):
            return f"无法解析时间: {raw}"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=src)
    return parsed.astimezone(dst).isoformat(timespec="seconds")


if __name__ == "__main__":
    mcp.run()
