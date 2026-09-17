"""n8n 任务推送：未配置则跳过。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[2]
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SRC))


class N8nTaskTest(unittest.TestCase):
    def test_skip_when_url_empty(self) -> None:
        from algorithm.n8n_task import push_companion_task

        with patch("algorithm.n8n_task.ensure_uvgraph_root"):
            with patch("src.configs.get_app_settings") as settings:
                with patch("src.configs.get_yaml_config", return_value={"n8n": {"enabled": True, "webhook_url": ""}}):
                    settings.return_value.n8n_webhook_url = ""
                    settings.return_value.n8n_webhook_secret = ""
                    out = push_companion_task(kind="remind", title="抽血")
        self.assertTrue(out.get("ok"))
        self.assertEqual(out.get("skipped"), "n8n_not_configured")


if __name__ == "__main__":
    unittest.main()
