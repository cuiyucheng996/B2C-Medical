from pathlib import Path

from configs.model_paths import CHECKPOINT_ROOT, MODEL_ROOT, PRETRAINED_ROOT
from configs.paths_b2c import B2C_ROOT

ROOT_DIR = B2C_ROOT
DATA_DIR = ROOT_DIR / "data"
PRE_TRAINED_DIR = PRETRAINED_ROOT
CHECKPOINT_DIR = CHECKPOINT_ROOT

# 兼容旧 import
__all__ = ["ROOT_DIR", "DATA_DIR", "PRE_TRAINED_DIR", "CHECKPOINT_DIR", "MODEL_ROOT"]
