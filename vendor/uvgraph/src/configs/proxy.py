"""代理与运行时环境：从 .env / config.yaml 加载，供 Git / httpx / LLaMA Factory 使用。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_YAML = CONFIG_DIR / "config.yaml"
DEFAULT_VENV_SCRIPTS = PROJECT_ROOT / ".venv" / "Scripts"

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_RUNTIME_ENV_KEYS = (
    "PYTHONUTF8",
    "PYTHONIOENCODING",
    "LLAMAFACTORY_DATA_DIR",
    "QWEN25_INSTRUCT_MODEL_PATH",
)

__all__ = [
    "ENV_FILE",
    "PROJECT_ROOT",
    "apply_project_env",
    "apply_proxy_env",
    "ensure_local_no_proxy",
    "get_git_email",
    "get_git_password",
    "get_git_remote_url",
    "get_git_username",
    "get_llamafactory_data_dir",
    "get_proxy_url",
    "get_qwen25_instruct_model_path",
    "httpx_async_client_kwargs",
    "httpx_client_kwargs",
    "read_proxy_url_from_env",
]


def _read_env_values() -> dict[str, str | None]:
    if not ENV_FILE.is_file():
        return {}
    return dotenv_values(ENV_FILE)


def _read_git_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        return {}
    with CONFIG_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    git_cfg = data.get("git") if isinstance(data, dict) else None
    return git_cfg if isinstance(git_cfg, dict) else {}


def read_proxy_url_from_env() -> str | None:
    """从 .env 读取代理地址（不写入 os.environ）。"""
    values = _read_env_values()
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = values.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


_LOCAL_NO_PROXY = ("localhost", "127.0.0.1", "::1")


def ensure_local_no_proxy() -> None:
    """本机服务（Milvus / Neo4j / Redis）不走 HTTP 代理。"""
    for key in ("NO_PROXY", "no_proxy"):
        parts = [p.strip() for p in os.environ.get(key, "").split(",") if p.strip()]
        for host in _LOCAL_NO_PROXY:
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def apply_proxy_env() -> None:
    """将 .env 中的 HTTP(S)_PROXY / ALL_PROXY 写入 os.environ（供 Hub / Git 等显式调用）。"""
    proxy = read_proxy_url_from_env()
    if proxy:
        for key in _PROXY_ENV_KEYS:
            os.environ[key] = proxy
            os.environ[key.lower()] = proxy
    ensure_local_no_proxy()


def _prepend_path(path: Path) -> None:
    if not path.is_dir():
        return
    prefix = str(path)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if prefix not in parts:
        os.environ["PATH"] = prefix + (os.pathsep + current if current else "")


def apply_project_env() -> None:
    """加载 .env 中的代理、UTF-8 与 LLaMA Factory 运行环境。"""
    values = _read_env_values()
    apply_proxy_env()

    for key in _RUNTIME_ENV_KEYS:
        value = values.get(key)
        if value:
            os.environ[key] = str(value).strip()

    scripts = values.get("LLAMAFACTORY_VENV_SCRIPTS")
    scripts_path = Path(str(scripts).strip()) if scripts else DEFAULT_VENV_SCRIPTS
    _prepend_path(scripts_path)


def get_proxy_url() -> str | None:
    """返回配置的代理 URL（优先读 .env 文件）。"""
    return read_proxy_url_from_env() or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")


def httpx_client_kwargs(*, use_proxy: bool = True, **extra: Any) -> dict[str, Any]:
    """构造 httpx.Client 参数。``use_proxy=False`` 时强制直连。"""
    kwargs: dict[str, Any] = {"trust_env": False, **extra}
    if use_proxy:
        proxy = get_proxy_url()
        if proxy:
            kwargs.setdefault("proxy", proxy)
    return kwargs


def httpx_async_client_kwargs(*, use_proxy: bool = True, **extra: Any) -> dict[str, Any]:
    """构造 httpx.AsyncClient 参数。"""
    return httpx_client_kwargs(use_proxy=use_proxy, **extra)


def get_git_remote_url() -> str | None:
    """远程地址：优先 .env GIT_REMOTE_URL，否则 config.yaml git.remote_url。"""
    values = _read_env_values()
    env_url = values.get("GIT_REMOTE_URL")
    if env_url:
        return str(env_url).strip()
    yaml_url = _read_git_yaml().get("remote_url")
    if yaml_url:
        return str(yaml_url).strip()
    return None


def get_git_username() -> str | None:
    """GitHub 用户名：优先 .env GIT_USERNAME，否则 config.yaml git.username。"""
    values = _read_env_values()
    env_user = values.get("GIT_USERNAME")
    if env_user:
        return str(env_user).strip()
    yaml_user = _read_git_yaml().get("username")
    if yaml_user:
        return str(yaml_user).strip()
    return None


def get_git_email() -> str | None:
    """提交邮箱：优先 .env GIT_EMAIL，否则 config.yaml git.email。"""
    values = _read_env_values()
    env_email = values.get("GIT_EMAIL")
    if env_email:
        return str(env_email).strip()
    yaml_email = _read_git_yaml().get("email")
    if yaml_email:
        return str(yaml_email).strip()
    username = get_git_username()
    if username:
        return f"{username}@users.noreply.github.com"
    return None


def get_git_password() -> str | None:
    """GitHub 密码或 PAT，仅从 .env 的 GIT_PASSWORD 读取。"""
    values = _read_env_values()
    value = values.get("GIT_PASSWORD")
    if not value:
        return None
    return str(value).strip()


def get_llamafactory_data_dir() -> str:
    apply_project_env()
    return os.environ.get("LLAMAFACTORY_DATA_DIR", "train/llamafactory_data")


def get_qwen25_instruct_model_path() -> str | None:
    apply_project_env()
    return os.environ.get("QWEN25_INSTRUCT_MODEL_PATH")
