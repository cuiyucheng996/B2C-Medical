"""模型与 checkpoint 路径（相对 B2C 项目根）。"""

from __future__ import annotations

from pathlib import Path

from configs.paths_b2c import B2C_ROOT

MODEL_ROOT = B2C_ROOT / "pretrained"
PRETRAINED_ROOT = MODEL_ROOT
CHECKPOINT_ROOT = B2C_ROOT / "checkpoint"
INTENT_MAJOR_WEIGHTS = CHECKPOINT_ROOT / "intent_major" / "checkpoint-184"

__all__ = [
    "MODEL_ROOT",
    "PRETRAINED_ROOT",
    "CHECKPOINT_ROOT",
    "INTENT_MAJOR_WEIGHTS",
]
