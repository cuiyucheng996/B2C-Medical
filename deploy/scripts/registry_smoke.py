"""检查 Consul/Nacos 是否可达，并打印 b2c-companion 解析结果。"""

from __future__ import annotations

import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[2]
B2C_SRC = B2C_ROOT / "src"
if str(B2C_SRC) not in sys.path:
    sys.path.insert(0, str(B2C_SRC))
if str(B2C_ROOT) not in sys.path:
    sys.path.insert(0, str(B2C_ROOT))

from dotenv import load_dotenv

from configs.paths_b2c import ensure_b2c_runtime_path

load_dotenv(B2C_ROOT / ".env")
ensure_b2c_runtime_path()

from infra.registry.discovery import companion_base_url
from infra.registry.settings import consul_addr, discovery_backends, nacos_addr


def main() -> None:
    print("B2C_SERVICE_DISCOVERY =", ",".join(sorted(discovery_backends())) or "(off)")
    print("CONSUL_HTTP_ADDR =", consul_addr())
    print("NACOS_SERVER_ADDR =", nacos_addr())
    print("companion_base_url =", companion_base_url())


if __name__ == "__main__":
    main()
