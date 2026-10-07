"""json_sanitize 与工具信封校验单测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from algorithm.json_sanitize import extract_json_object_text, loads_json_dict  # noqa: E402
from tools.tool_envelope import dump_envelope, validate_envelope_json  # noqa: E402


def test_fence_and_extract() -> None:
    raw = '说明\n```json\n{"fit":true,"action":"compose","reason":"x"}\n```'
    blob = extract_json_object_text(raw)
    row = loads_json_dict(blob)
    assert row["fit"] is True


def test_validate_envelope_rejects_missing_field() -> None:
    bad = json.dumps({"ok": True, "tool": "t", "usable": True})
    fixed = validate_envelope_json(bad, tool="t")
    data = json.loads(fixed)
    assert data["usable"] is False
    assert data["ok"] is False


def test_dump_envelope_validates_shape() -> None:
    text = dump_envelope(tool="lookup_stated_consult", ok=True, summary="hi", data={})
    data = json.loads(text)
    assert data["tool"] == "lookup_stated_consult"
    assert "retryable" in data


if __name__ == "__main__":
    test_fence_and_extract()
    test_validate_envelope_rejects_missing_field()
    test_dump_envelope_validates_shape()
    print("ok")
