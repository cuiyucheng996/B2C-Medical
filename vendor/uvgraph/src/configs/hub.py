"""项目模型 Hub：统一走本项目 HuggingFace / ModelScope 缓存与代理。

下载前确保：
- ``apply_proxy_env``（.env 代理）
- ``apply_cache_env``（``.cache/huggingface`` / ``.cache/modelscope``）

配置见 ``config.yaml`` 的 ``hub`` 段：
- ``source``: ``huggingface`` | ``modelscope`` | ``auto``（先 HF 失败再 MS）
- ``hf_endpoint``: 可选镜像，如 ``https://hf-mirror.com``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml

from .cache import (
    HF_HUB_CACHE_DIR,
    MODELSCOPE_CACHE_DIR,
    apply_cache_env,
    cache_env_applied,
)
from .proxy import apply_proxy_env

logger = logging.getLogger(__name__)

CONFIG_YAML = Path(__file__).parent / "config.yaml"
HubSource = Literal["huggingface", "modelscope", "auto"]

__all__ = [
    "HubSource",
    "ensure_hub_env",
    "get_hub_settings",
    "hub_local_repo_dir",
    "hub_snapshot_download",
]


def _read_hub_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        return {}
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    hub = data.get("hub") if isinstance(data, dict) else None
    return hub if isinstance(hub, dict) else {}


def get_hub_settings() -> dict[str, Any]:
    """读取 hub 配置；可用环境变量覆盖。"""
    cfg = _read_hub_yaml()
    source = str(
        os.environ.get("MODEL_HUB_SOURCE")
        or cfg.get("source")
        or "huggingface"
    ).strip().lower()
    if source not in ("huggingface", "modelscope", "auto"):
        source = "huggingface"
    endpoint = (
        os.environ.get("HF_ENDPOINT")
        or os.environ.get("HUGGINGFACE_HUB_ENDPOINT")
        or cfg.get("hf_endpoint")
        or ""
    )
    endpoint = str(endpoint).strip().rstrip("/")
    return {
        "source": source,
        "hf_endpoint": endpoint or None,
    }


def ensure_hub_env() -> None:
    """下载前调用：代理 + 项目缓存目录 + 可选 HF 镜像。"""
    apply_proxy_env()
    if not cache_env_applied():
        apply_cache_env()
    settings = get_hub_settings()
    endpoint = settings.get("hf_endpoint")
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
        # huggingface_hub 部分版本读这个
        os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", str(endpoint))


def _hf_repo_cache_dir(repo_id: str) -> Path:
    # HuggingFace hub 布局：models--org--name
    safe = "models--" + repo_id.replace("/", "--")
    return HF_HUB_CACHE_DIR / safe


def hub_local_repo_dir(repo_id: str) -> Path | None:
    """若本地 HF hub 已有该 repo 的 snapshots，返回最新 snapshot 目录。"""
    hub = _hf_repo_cache_dir(repo_id)
    snaps = hub / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # 优先 refs/main 指向的 commit
    ref_main = hub / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        target = snaps / commit
        if target.is_dir():
            return target
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _download_huggingface(
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "cache_dir": str(HF_HUB_CACHE_DIR),
        "local_files_only": local_files_only,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    path = snapshot_download(**kwargs)
    return Path(path)


def _download_modelscope(
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except ImportError as exc:
        raise ImportError(
            "ModelScope 未安装：请执行 `uv add modelscope`，或将 hub.source 改为 huggingface"
        ) from exc

    MODELSCOPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "model_id": repo_id,
        "cache_dir": str(MODELSCOPE_CACHE_DIR),
    }
    # modelscope 新版本支持 allow_patterns；旧版本忽略
    if allow_patterns:
        try:
            path = ms_download(**kwargs, allow_patterns=allow_patterns)
        except TypeError:
            path = ms_download(**kwargs)
    else:
        path = ms_download(**kwargs)
    return Path(path)


def hub_snapshot_download(
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
    prefer_local: bool = True,
) -> Path:
    """下载/定位模型仓库到本项目缓存。

    Args:
        repo_id: 如 ``InfiniFlow/deepdoc``
        allow_patterns: HF allow_patterns（MS 尽力透传）
        local_files_only: 仅本地（HF）
        prefer_local: 先查本地 HF hub，命中则不联网
    """
    ensure_hub_env()
    settings = get_hub_settings()
    source: HubSource = settings["source"]

    if prefer_local and not local_files_only:
        local = hub_local_repo_dir(repo_id)
        if local is not None:
            logger.info("hub local hit: %s -> %s", repo_id, local)
            return local

    errors: list[str] = []

    def try_hf() -> Path:
        logger.info(
            "hub download [huggingface] %s -> %s",
            repo_id,
            HF_HUB_CACHE_DIR,
        )
        return _download_huggingface(
            repo_id,
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
        )

    def try_ms() -> Path:
        logger.info(
            "hub download [modelscope] %s -> %s",
            repo_id,
            MODELSCOPE_CACHE_DIR,
        )
        return _download_modelscope(repo_id, allow_patterns=allow_patterns)

    if source == "huggingface":
        return try_hf()
    if source == "modelscope":
        return try_ms()

    # auto：先 HF，失败再 MS
    try:
        return try_hf()
    except Exception as exc:
        errors.append(f"huggingface: {exc}")
        logger.warning("HF download failed for %s: %s; try modelscope", repo_id, exc)
    try:
        return try_ms()
    except Exception as exc:
        errors.append(f"modelscope: {exc}")
        raise RuntimeError(
            f"无法下载 {repo_id}（已走项目 hub 缓存/代理）。详情: {'; '.join(errors)}"
        ) from exc
