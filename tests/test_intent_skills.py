"""intent_skills 与 intent_catalog 对齐校验。"""

from __future__ import annotations

import sys
from pathlib import Path

_B2C_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_B2C_SRC) not in sys.path:
    sys.path.insert(0, str(_B2C_SRC))

from algorithm.intent_manifest import load_intent_manifest  # noqa: E402
from prompt.intent_skill_loader import load_intent_skill  # noqa: E402


def test_every_catalog_handler_has_skill() -> None:
    manifest = load_intent_manifest()
    missing: list[str] = []
    for row in manifest.rows:
        handler = str(row.get("handler") or "").strip()
        if not handler:
            continue
        if not load_intent_skill(handler):
            missing.append(handler)
    import yaml
    from configs.paths_b2c import B2C_ROOT

    data = yaml.safe_load((B2C_ROOT / "data" / "intent_catalog.yaml").read_text(encoding="utf-8"))
    for h in data.get("extra_bind_handlers") or []:
        hs = str(h).strip()
        if hs and not load_intent_skill(hs):
            missing.append(hs)
    assert not missing, f"缺少 intent_skills：{missing}"


if __name__ == "__main__":
    test_every_catalog_handler_has_skill()
    print("ok")
