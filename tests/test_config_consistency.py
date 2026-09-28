"""配置单一源：intent_catalog.yaml、MCP YAML、INTENT_BIND 不漂移。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("COMPANION_CHECKPOINTER", "memory")

_SRC = Path(__file__).resolve().parents[1] / "src"
_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))


class ConfigConsistencyTest(unittest.TestCase):
    def test_intent_manifest_valid(self) -> None:
        from algorithm.intent_manifest import load_intent_manifest, validate_intent_manifest

        self.assertEqual(validate_intent_manifest(), [])
        manifest = load_intent_manifest()
        self.assertGreaterEqual(len(manifest.rows), 20)
        self.assertEqual(len(manifest.by_name), len(manifest.rows))

    def test_intent_rules_reference_catalog(self) -> None:
        from algorithm.intent_catalog import INTENT_BY_NAME
        from algorithm.intent_rule import INTENT_RULES

        for intent, _kw in INTENT_RULES:
            self.assertIn(intent, INTENT_BY_NAME, msg=intent)

    def test_intent_bind_handlers_in_catalog(self) -> None:
        from algorithm.intent_manifest import load_intent_manifest
        from tools.setup import INTENT_BIND

        handlers = load_intent_manifest().handlers
        for (major, handler), _tools in INTENT_BIND.items():
            if not handler:
                continue
            self.assertIn(handler, handlers, msg=f"{major}/{handler}")

    def test_mcp_yaml_matches_registry(self) -> None:
        from tools.mcp_yaml_registry import validate_mcp_yaml_registry

        issues = validate_mcp_yaml_registry()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_train_templates_intents_in_catalog(self) -> None:
        from algorithm.intent_manifest import load_intent_manifest

        manifest = load_intent_manifest()
        for intent in manifest.train_templates:
            self.assertIn(intent, manifest.by_name)


if __name__ == "__main__":
    unittest.main()
