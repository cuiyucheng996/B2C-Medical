"""各 FastAPI 子服务统一挂载 OpenTelemetry。"""

from __future__ import annotations

from typing import Any


def mount_otel(app: Any, service_name: str) -> None:
    try:
        from algorithm.otel_setup import setup_otel

        setup_otel(app=app, service_name=service_name)
    except Exception:
        pass
