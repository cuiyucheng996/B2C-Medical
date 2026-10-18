"""读取 middleware_third_party 内 config.yaml 与 sources.yaml。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from configs.paths_b2c import B2C_ROOT, ensure_b2c_runtime_path

ensure_b2c_runtime_path()

DEFAULT_DATA_DIR = "server/builtin_server/directed_crawl/crawl_data"


def _yaml_block() -> dict[str, Any]:
    from src.configs import get_yaml_config

    raw = get_yaml_config().get("directed_crawl")
    return raw if isinstance(raw, dict) else {}


def sources_path() -> Path:
    from configs.data_paths import DIRECTED_CRAWL_SOURCES

    rel = str(_yaml_block().get("sources_file") or "").strip()
    if rel:
        candidate = (B2C_ROOT / rel).resolve()
        if candidate.is_file():
            return candidate
    return DIRECTED_CRAWL_SOURCES.resolve()


def load_sources() -> dict[str, Any]:
    import yaml

    path = sources_path()
    if not path.is_file():
        return {"whitelist_hosts": [], "rpa_output_dirs": [], "profiles": {}, "seed_urls": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def cache_root() -> Path:
    rel = str(_yaml_block().get("data_dir") or DEFAULT_DATA_DIR).strip()
    root = (B2C_ROOT / rel).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("jobs", "inbox", "pages", "excel"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def max_fetch_bytes() -> int:
    try:
        return int(_yaml_block().get("max_fetch_bytes") or 400_000)
    except (TypeError, ValueError):
        return 400_000


def host_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    hosts = [str(h).lower() for h in load_sources().get("whitelist_hosts") or []]
    return any(host == h or host.endswith("." + h) for h in hosts if h)


def excel_path_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    dirs = load_sources().get("rpa_output_dirs") or []
    for item in dirs:
        base = Path(str(item).strip())
        if not base.is_absolute():
            base = (B2C_ROOT / base).resolve()
        else:
            base = base.resolve()
        try:
            resolved.relative_to(base)
            return True
        except ValueError:
            continue
    return False
