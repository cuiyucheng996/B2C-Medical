"""UIE 模型缓存：统一落到项目 .cache/uie。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from src.configs.cache import UIE_CACHE_DIR, ensure_cache_dirs

__all__ = [
    "UIE_CODE_ROOT",
    "ensure_uie_model",
    "uie_cache_cwd",
    "uie_paddle_model_dir",
    "uie_pytorch_model_dir",
]

# B2C business/External/uie_pytorch（与主项目 B2B 目录解耦）
UIE_CODE_ROOT = Path(__file__).resolve().parents[4] / "External" / "uie_pytorch"


def uie_paddle_model_dir(model: str) -> Path:
    return UIE_CACHE_DIR / model


def uie_pytorch_model_dir(model: str) -> Path:
    return UIE_CACHE_DIR / f"{model.replace('-', '_')}_pytorch"


@contextmanager
def uie_cache_cwd():
    """在 .cache/uie 下执行 UIE 下载/转换（uie_predictor 使用相对路径）。"""
    ensure_cache_dirs()
    UIE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(UIE_CACHE_DIR)
    try:
        yield UIE_CACHE_DIR
    finally:
        os.chdir(previous)


def ensure_uie_model(model: str = "uie-base", *, force: bool = False) -> Path:
    """下载 Paddle 权重并转换为 PyTorch，返回 pytorch 模型目录。"""
    if str(UIE_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(UIE_CODE_ROOT))

    pytorch_dir = uie_pytorch_model_dir(model)
    if (
        not force
        and pytorch_dir.is_dir()
        and (pytorch_dir / "pytorch_model.bin").is_file()
    ):
        return pytorch_dir

    with uie_cache_cwd():
        from convert import check_model, extract_and_convert

        paddle_name = model
        pytorch_name = f"{model.replace('-', '_')}_pytorch"
        check_model(paddle_name)
        extract_and_convert(paddle_name, pytorch_name, verbose=True)

    if not (pytorch_dir / "pytorch_model.bin").is_file():
        raise RuntimeError(f"UIE 模型转换失败，未找到 {pytorch_dir / 'pytorch_model.bin'}")
    return pytorch_dir
