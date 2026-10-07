"""UIE 信息抽取工具。"""

from .model_cache import (
    UIE_CODE_ROOT,
    ensure_uie_model,
    uie_cache_cwd,
    uie_paddle_model_dir,
    uie_pytorch_model_dir,
)

__all__ = [
    "UIE_CODE_ROOT",
    "ensure_uie_model",
    "uie_cache_cwd",
    "uie_paddle_model_dir",
    "uie_pytorch_model_dir",
]
