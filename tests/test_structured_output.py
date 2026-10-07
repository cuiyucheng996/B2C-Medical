"""structured_block 与 parse_json_with_schema 单测。"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.structured_output import parse_json_with_schema  # noqa: E402
from prompt.structured_block import compose_structured_system_prompt  # noqa: E402
from schema.llm_outputs import CompanionThinkOutput  # noqa: E402


def test_schema_block_at_end() -> None:
    text = compose_structured_system_prompt("上文说明", CompanionThinkOutput, examples=None)
    assert text.startswith("上文说明")
    assert text.rstrip().endswith("}")


def test_parse_think_json() -> None:
    row = parse_json_with_schema(
        '{"fit":true,"action":"compose","reason":"ok"}',
        CompanionThinkOutput,
    )
    assert row.fit is True
    assert row.action == "compose"


if __name__ == "__main__":
    test_schema_block_at_end()
    test_parse_think_json()
    print("ok")
