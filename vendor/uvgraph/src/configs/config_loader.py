"""项目配置：YAML 默认值 + .env 密钥，统一加载入口。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from .cache import (
    CACHE_ROOT,
    CHROMA_CACHE_DIR,
    MINERU_OUTPUT_DIR,
    MODELSCOPE_CACHE_DIR,
    apply_cache_env,
    configure_cache_paths,
    configure_external_tool_paths,
)
from .hub import ensure_hub_env
from .proxy import ensure_local_no_proxy

CONFIG_DIR = Path(__file__).parent
# B2C 独立仓：配置与 .env 在 B2C business 根目录（vendor/uvgraph/src/configs → parents[4]）
B2C_BUSINESS_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = (
    B2C_BUSINESS_ROOT
    / "data"
    / "dev_resources"
    / "middleware_third_party"
    / "config.yaml"
)
PROJECT_ROOT = B2C_BUSINESS_ROOT
ENV_FILE = B2C_BUSINESS_ROOT / ".env"

AppEnv = Literal["development", "staging", "production"]
EmbeddingProvider = Literal["dashscope", "huggingface"]
EmbeddingDevice = Literal["cpu", "cuda"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

REQUIRED_SECTIONS: dict[str, list[str]] = {
    "mysql": ["host", "user", "password", "database", "charset", "port"],
    "milvus": ["uri", "token", "db_name"],
    "redis": ["host", "port", "db", "password", "faq_redis_namespace", "medical_faq_redis_namespace"],
}

# 可选段：存在即提取，不做必填校验（neo4j 段可整体缺失）
OPTIONAL_INFRA_SECTIONS: tuple[str, ...] = ("neo4j", "ragflow")


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {config_path} 文件内容格式错误")
    return data


def _require_section(data: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        raise ValueError(f"{section} 缺少配置，或者配置格式错误")
    return section_data


def _require_keys(section_data: dict[str, Any], section: str) -> dict[str, Any]:
    required = REQUIRED_SECTIONS[section]
    missing_keys = [key for key in required if key not in section_data]
    if missing_keys:
        raise ValueError(f"{section} 缺少相关 key: {missing_keys}")
    return section_data


def _normalize_redis(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    password = normalized.get("password")
    if password in ("None", ""):
        normalized["password"] = None
    return normalized


def _section_dict(data: dict[str, Any], section: str) -> dict[str, Any]:
    section_data = data.get(section)
    if not isinstance(section_data, dict):
        return {}
    return {k: v for k, v in section_data.items() if v is not None}


def extract_infra_config(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """提取并校验 mysql / milvus / redis 配置（neo4j 为可选段）。"""
    infra: dict[str, dict[str, Any]] = {}
    for section in REQUIRED_SECTIONS:
        section_data = _require_section(data, section)
        infra[section] = _require_keys(section_data, section)

    infra["redis"] = _normalize_redis(infra["redis"])
    for section in OPTIONAL_INFRA_SECTIONS:
        infra[section] = _section_dict(data, section)
    return infra


def parse_yaml_app_defaults(data: dict[str, Any]) -> dict[str, Any]:
    """将 yaml 中非敏感项映射为 AppSettings 字段默认值。"""
    app = _section_dict(data, "app")
    embedding = _section_dict(data, "embedding")
    llm = _section_dict(data, "llm")
    deepseek = llm.get("deepseek") if isinstance(llm.get("deepseek"), dict) else {}
    openai = llm.get("openai") if isinstance(llm.get("openai"), dict) else {}
    kimi = llm.get("kimi") if isinstance(llm.get("kimi"), dict) else {}
    think = llm.get("think") if isinstance(llm.get("think"), dict) else {}

    openai_base_url = openai.get("base_url")
    return {
        "app_env": app.get("env", "development"),
        "log_level": app.get("log_level", "INFO"),
        "embedding_provider": embedding.get("provider", "huggingface"),
        "embedding_model": embedding.get("model", "BAAI/bge-small-zh-v1.5"),
        "embedding_device": embedding.get("device", "cpu"),  # yaml embedding.device；改 yaml 需进程重载
        "deepseek_base_url": deepseek.get("base_url", "https://api.deepseek.com/v1"),
        "deepseek_model": deepseek.get("model", "deepseek-chat"),
        "openai_base_url": openai_base_url or None,
        "openai_model": openai.get("model", "gpt-4o-mini"),
        "kimi_base_url": kimi.get("base_url", "https://api.moonshot.cn/v1"),
        "kimi_model": kimi.get("model", "moonshot-v1-8k"),
        "thinking_base_url": think.get("base_url")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "thinking_model": think.get("model") or "qwen-plus",
        "thinking_enabled": think.get("enabled", True) is not False,
    }


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sync_mineru_tools_config(mineru: dict[str, Any]) -> Path | None:
    """将 config.yaml 的 mineru 路径同步到 MinerU 的 mineru.json。"""
    tools_config = mineru.get("tools_config")
    if not tools_config:
        return None
    config_path = _resolve_project_path(tools_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"config_version": "1.3.2"}

    models_dir = mineru.get("models_dir")
    if isinstance(models_dir, dict) and models_dir:
        payload["models-dir"] = {
            key: str(_resolve_project_path(path))
            for key, path in models_dir.items()
        }

    model_source = mineru.get("model_source")
    if model_source:
        payload["model-source"] = str(model_source)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
        f.write("\n")
    return config_path


def _apply_mineru_env(data: dict[str, Any]) -> None:
    mineru = _section_dict(data, "mineru")
    if not mineru:
        return

    config_path = _sync_mineru_tools_config(mineru)
    if config_path is not None:
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)

    model_source = mineru.get("model_source")
    if model_source:
        os.environ["MINERU_MODEL_SOURCE"] = str(model_source)

    device = mineru.get("device")
    if device:
        os.environ["MINERU_DEVICE_MODE"] = str(device)

    modelscope_cache = mineru.get("modelscope_cache")
    if modelscope_cache:
        os.environ["MODELSCOPE_CACHE"] = str(_resolve_project_path(modelscope_cache))

    output_dir = mineru.get("output_dir")
    if output_dir:
        out_path = _resolve_project_path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        os.environ["MINERU_OUTPUT_DIR"] = str(out_path)


def _bootstrap() -> dict[str, Any]:
    # 不在启动时全局注入 HTTP_PROXY，避免国内 LLM 被误导向未启动的本地代理。
    # Hub / Git 等需要代理的场景在各自模块内显式调用 apply_proxy_env()。
    data = load_yaml_config()
    cache_root = _section_dict(data, "cache").get("root", ".cache") or ".cache"
    configure_cache_paths(
        _resolve_project_path(cache_root),
        project_root=B2C_BUSINESS_ROOT,
    )
    tools = _section_dict(data, "external_tools")
    tess = tools.get("tesseract_dir")
    pop = tools.get("poppler_dir")
    configure_external_tool_paths(
        tesseract_dir=str(_resolve_project_path(tess)) if tess else None,
        poppler_dir=str(_resolve_project_path(pop)) if pop else None,
    )
    apply_cache_env()
    ensure_local_no_proxy()
    # hub：HF 镜像 / MODEL_HUB_SOURCE 等，供 parser 等模型下载
    ensure_hub_env()
    _apply_mineru_env(data)
    return data


_yaml_config = _bootstrap()
YAML_APP_DEFAULTS = parse_yaml_app_defaults(_yaml_config)
_D = YAML_APP_DEFAULTS

_infra_config = extract_infra_config(_yaml_config)
_vector_store = _yaml_config.get("vector_store") if isinstance(_yaml_config.get("vector_store"), dict) else {}
VECTOR_STORE_BACKEND: str = str(
    os.getenv("VECTOR_STORE_BACKEND") or _vector_store.get("backend") or "chroma"
).strip().lower()
_chroma_section = _yaml_config.get("chroma") if isinstance(_yaml_config.get("chroma"), dict) else {}
CHROMA_VECTOR_CONFIG: dict[str, Any] = dict(_chroma_section)
_chroma_persist = str(CHROMA_VECTOR_CONFIG.get("persist_directory") or "").strip()
if _chroma_persist:
    CHROMA_VECTOR_CONFIG["persist_directory"] = str(_resolve_project_path(_chroma_persist))
else:
    CHROMA_VECTOR_CONFIG["persist_directory"] = str(CHROMA_CACHE_DIR)
MYSQL_CONFIG: dict[str, Any] = _infra_config["mysql"]
MYSQL_INGEST_DATABASE: str = (
    _infra_config["mysql"].get("ingest_database") or _infra_config["mysql"]["database"]
)
MILVUS_CONFIG: dict[str, Any] = _infra_config["milvus"]
REDIS_CONFIG: dict[str, Any] = _infra_config["redis"]
NEO4J_CONFIG: dict[str, Any] = _infra_config["neo4j"]
RAGFLOW_CONFIG: dict[str, Any] = _infra_config.get("ragflow") or {}


class AppSettings(BaseSettings):
    """密钥从 .env 读取；非敏感默认值来自 config.yaml。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: AppEnv = _D["app_env"]  # type: ignore[assignment]
    log_level: LogLevel = _D["log_level"]  # type: ignore[assignment]

    dashscope_api_key: str = ""

    embedding_provider: EmbeddingProvider = _D["embedding_provider"]  # type: ignore[assignment]
    embedding_model: str = _D["embedding_model"]
    embedding_device: EmbeddingDevice = _D["embedding_device"]  # type: ignore[assignment]

    deepseek_api_key: str = ""
    deepseek_base_url: str = _D["deepseek_base_url"]
    deepseek_model: str = _D["deepseek_model"]

    openai_api_key: str = ""
    openai_base_url: str | None = _D["openai_base_url"]
    openai_model: str = _D["openai_model"]

    kimi_api_key: str = ""
    kimi_base_url: str = _D["kimi_base_url"]
    kimi_model: str = _D["kimi_model"]

    thinking_api_key: str = ""
    thinking_base_url: str = _D["thinking_base_url"]
    thinking_model: str = _D["thinking_model"]
    thinking_enabled: bool = _D["thinking_enabled"]

    # Neo4j 密码（仅放 .env，非敏感项见 config.yaml neo4j 段）
    neo4j_password: str = ""
    ragflow_api_key: str = ""
    ragflow_email: str = ""
    ragflow_password: str = ""
    ragflow_admin_email: str = ""
    ragflow_admin_password: str = ""
    n8n_webhook_url: str = ""
    n8n_webhook_secret: str = ""

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings()


app_settings = get_app_settings()


def get_yaml_config() -> dict[str, Any]:
    return _yaml_config



#对外暴露的变量
__all__ = [
    #缓存相关
    "CACHE_ROOT",
    "CHROMA_CACHE_DIR",
    "MINERU_OUTPUT_DIR",
    "MODELSCOPE_CACHE_DIR",
    "CONFIG_PATH",  #配置文件路径，config.yaml 在哪
    "ENV_FILE",
    #数据库相关
    "CHROMA_VECTOR_CONFIG",
    "VECTOR_STORE_BACKEND",
    "MILVUS_CONFIG", #milvus配置
    "MYSQL_CONFIG", #mysql配置
    "MYSQL_INGEST_DATABASE", # markdown 入库默认库名
    "REDIS_CONFIG", #redis配置
    "NEO4J_CONFIG", #neo4j配置
    "RAGFLOW_CONFIG",
    #项目根目录
    "PROJECT_ROOT", #项目根目录     
    # 应用设置相关
    "YAML_APP_DEFAULTS",  # dict：仅 yaml 非敏感项默认值
    "AppSettings",  # 类：yaml 默认值 + .env 密钥
    "parse_yaml_app_defaults",  # 函数：yaml → 扁平 dict
    "get_app_settings",  # 获取 AppSettings 单例
    "app_settings",  # 模块级 AppSettings 实例
    # 配置文件相关
    "get_yaml_config", #获取yaml配置，启动时已经读好的 YAML 内容
    "load_yaml_config", #加载yaml配置，读取 config.yaml 内容
]
