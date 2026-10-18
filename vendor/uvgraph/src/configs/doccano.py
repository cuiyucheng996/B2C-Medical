"""Doccano 标注服务路径与 UIE 标签配置（读 config.yaml + .env 密钥）。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config_loader import PROJECT_ROOT, get_yaml_config

__all__ = ["get_doccano_config", "resolve_doccano_path"]


def _section_dict(data: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return {}
    return {k: v for k, v in section_data.items() if v is not None}


def resolve_doccano_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache
def get_doccano_config() -> dict[str, Any]:
    """返回 Doccano 服务 URL、数据目录与 UIE 标签配置。"""
    cfg = _section_dict(get_yaml_config(), "doccano")
    project = cfg.get("project") if isinstance(cfg.get("project"), dict) else {}

    port = int(os.getenv("DOCCANO_PORT", cfg.get("port", 8088)))
    host = str(cfg.get("host", "localhost"))
    url = str(cfg.get("url") or f"http://{host}:{port}")

    samples_dir = resolve_doccano_path(cfg.get("samples_dir", "model/datasets/uie/doccano/samples"))
    export_dir = resolve_doccano_path(cfg.get("export_dir", "model/datasets/uie/doccano/export"))
    uie_train_dir = resolve_doccano_path(cfg.get("uie_train_dir", "model/datasets/uie/train"))

    span_labels = cfg.get("span_labels") or ["部位", "疾病", "症状"]
    if not isinstance(span_labels, list):
        span_labels = ["部位", "疾病", "症状"]

    return {
        "host": host,
        "port": port,
        "url": url,
        "admin_username": os.getenv(
            "DOCCANO_ADMIN_USERNAME", str(cfg.get("admin_username", "admin"))
        ),
        "admin_password": os.getenv("DOCCANO_ADMIN_PASSWORD", ""),
        "admin_email": os.getenv(
            "DOCCANO_ADMIN_EMAIL", str(cfg.get("admin_email", "admin@example.com"))
        ),
        "samples_dir": samples_dir,
        "export_dir": export_dir,
        "uie_train_dir": uie_train_dir,
        "span_labels": [str(x) for x in span_labels],
        "project_type": str(project.get("type", "SequenceLabeling")),
        "upload_format": str(project.get("upload_format", "TextLine")),
        "allow_overlap": bool(project.get("allow_overlap", True)),
        "use_relation_labeling": bool(project.get("use_relation_labeling", False)),
        "docker_compose_dir": PROJECT_ROOT / "deploy" / "doccano",
    }
