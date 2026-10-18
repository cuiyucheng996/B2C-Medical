"""把共享配置发布到 Nacos 配置中心（Data ID: b2c-business.env）。"""

from __future__ import annotations

import argparse
import re
import urllib.parse
import urllib.request
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = B2C_ROOT / "data" / "dev_resources" / "middleware_third_party" / "middleware" / "nacos" / "b2c-business.env.example"
LIVE = B2C_ROOT / "data" / "dev_resources" / "middleware_third_party" / "middleware" / "nacos" / "b2c-business.env"
DOTENV = B2C_ROOT / ".env"
CONFIG_YAML = B2C_ROOT / "data" / "dev_resources" / "middleware_third_party" / "config.yaml"

CONTENT = """B2C_SERVICE_DISCOVERY=consul,nacos
B2C_NACOS_CONFIG=true
CONSUL_HTTP_ADDR=127.0.0.1:8500
NACOS_SERVER_ADDR=127.0.0.1:8848
B2C_SERVICE_HOST=127.0.0.1
B2C_SERVICE_REGISTER_HOST=host.docker.internal
B2C_CONSUL_HEALTH_HOST=host.docker.internal
B2C_COMPANION_SERVICE_NAME=b2c-companion
"""

_SHARED_PREFIXES = ("B2C_", "CONSUL_", "NACOS_", "REDIS_", "NEO4J_", "VECTOR_")
_SECRET_RE = re.compile(r"(_KEY|_SECRET|_PASSWORD|_TOKEN|_PIN)$", re.I)


def _addr() -> str:
    import os

    return (os.getenv("NACOS_SERVER_ADDR") or "127.0.0.1:8848").strip().rstrip("/")


def publish(content: str) -> None:
    data = urllib.parse.urlencode(
        {
            "dataId": "b2c-business.env",
            "group": "DEFAULT_GROUP",
            "tenant": "",
            "type": "text",
            "content": content,
        }
    ).encode("utf-8")
    req = urllib.request.Request(f"http://{_addr()}/nacos/v1/cs/configs", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        ok = resp.read().decode("utf-8", errors="replace").strip().lower()
        if ok not in {"true", "ok"}:
            raise SystemExit(f"Nacos 发布失败: {ok}")
    print("已发布 Nacos 配置: b2c-business.env (DEFAULT_GROUP, public)")


def _parse_env_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[7:].strip()
        if "=" not in raw:
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def _merge_maps(*maps: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for m in maps:
        merged.update(m)
    return merged


def _filter_dotenv(env: dict[str, str], *, with_secrets: bool) -> dict[str, str]:
    picked: dict[str, str] = {}
    for key, val in env.items():
        if not any(key.startswith(p) for p in _SHARED_PREFIXES):
            continue
        if not with_secrets and _SECRET_RE.search(key):
            continue
        if val:
            picked[key] = val
    return picked


def _yaml_gateway_ports() -> dict[str, str]:
    try:
        import yaml
    except ImportError:
        return {}
    if not CONFIG_YAML.is_file():
        return {}
    data = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    gateway = data.get("gateway")
    if not isinstance(gateway, dict):
        return {}
    ports = gateway.get("ports")
    if not isinstance(ports, dict):
        return {}
    mapping = {
        "companion": "B2C_PORT_COMPANION",
        "memory_rag": "B2C_PORT_MEMORY_RAG",
        "report_read": "B2C_PORT_REPORT_READ",
        "doc_ingest": "B2C_PORT_DOC_INGEST",
        "directed_crawl": "B2C_PORT_DIRECTED_CRAWL",
        "dev_vite": "B2C_PORT_DEV_VITE",
    }
    out: dict[str, str] = {}
    for name, env_key in mapping.items():
        val = ports.get(name)
        if isinstance(val, int):
            out[env_key] = str(val)
    public = gateway.get("public_url")
    if isinstance(public, str) and public.strip():
        out["B2C_GATEWAY_PUBLIC_URL"] = public.strip()
    return out


def build_content(*, merge_dotenv: bool, merge_config: bool, with_secrets: bool) -> str:
    base_text = CONTENT
    if LIVE.is_file():
        base_text = LIVE.read_text(encoding="utf-8")
    elif EXAMPLE.is_file():
        lines = [
            ln
            for ln in EXAMPLE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        base_text = "\n".join(lines) + ("\n" if lines else "")

    maps: list[dict[str, str]] = [_parse_env_lines(base_text)]
    if merge_config:
        maps.append(_yaml_gateway_ports())
    if merge_dotenv and DOTENV.is_file():
        maps.append(_filter_dotenv(_parse_env_lines(DOTENV.read_text(encoding="utf-8")), with_secrets=with_secrets))

    merged = _merge_maps(*maps)
    lines = [f"{k}={v}" for k, v in sorted(merged.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="发布 b2c-business.env 到 Nacos")
    parser.add_argument("--merge-dotenv", action="store_true", help="合并 .env 中 B2C_/CONSUL_/NACOS_ 等共享项")
    parser.add_argument("--merge-config", action="store_true", help="合并 config.yaml gateway.ports 为 B2C_PORT_*")
    parser.add_argument("--with-secrets", action="store_true", help="允许把 *_KEY 等写入 Nacos（生产慎用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印内容，不发布")
    args = parser.parse_args()

    body = build_content(
        merge_dotenv=args.merge_dotenv,
        merge_config=args.merge_config or args.merge_dotenv,
        with_secrets=args.with_secrets,
    )
    if args.dry_run:
        print(body)
        return
    publish(body)


if __name__ == "__main__":
    main()
