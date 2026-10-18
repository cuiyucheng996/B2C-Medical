"""项目本地缓存目录（统一 .cache）。"""

from __future__ import annotations

import os
from pathlib import Path

def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (
            ancestor
            / "data"
            / "dev_resources"
            / "middleware_third_party"
            / "config.yaml"
        ).is_file():
            return ancestor
    return here.parents[2]


PROJECT_ROOT = _discover_project_root()

CACHE_ROOT = PROJECT_ROOT / ".cache"
HF_CACHE_DIR = CACHE_ROOT / "huggingface"
HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "transformers"
SENTENCE_TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "sentence_transformers"
MODELSCOPE_CACHE_DIR = CACHE_ROOT / "modelscope"
MINERU_CACHE_DIR = CACHE_ROOT / "mineru"
MINERU_OUTPUT_DIR = MINERU_CACHE_DIR / "output"
TORCH_CACHE_DIR = CACHE_ROOT / "torch"
CHROMA_CACHE_DIR = CACHE_ROOT / "chroma"
UIE_CACHE_DIR = CACHE_ROOT / "uie"
EXTERNAL_TOOLS_DIR = CACHE_ROOT / "external"
DEFAULT_TESSERACT_DIR = EXTERNAL_TOOLS_DIR / "ocr"
DEFAULT_POPPLER_DIR = EXTERNAL_TOOLS_DIR / "poppler" / "poppler-24.08.0" / "Library" / "bin"

# 外部工具（Tesseract / Poppler）：手动复制到 .cache/external/，不进 Git；与 huggingface/hub 分离
TESSERACT_DIR = DEFAULT_TESSERACT_DIR
POPPLER_DIR: Path | None = None

_CACHE_ENV_APPLIED = False

__all__ = [
    "CACHE_ROOT",
    "CHROMA_CACHE_DIR",
    "DEFAULT_POPPLER_DIR",
    "DEFAULT_TESSERACT_DIR",
    "EXTERNAL_TOOLS_DIR",
    "HF_CACHE_DIR",
    "HF_HUB_CACHE_DIR",
    "MODELSCOPE_CACHE_DIR",
    "MINERU_CACHE_DIR",
    "MINERU_OUTPUT_DIR",
    "POPPLER_DIR",
    "PROJECT_ROOT",
    "SENTENCE_TRANSFORMERS_CACHE_DIR",
    "TESSERACT_DIR",
    "TORCH_CACHE_DIR",
    "TRANSFORMERS_CACHE_DIR",
    "UIE_CACHE_DIR",
    "apply_cache_env",
    "cache_env_applied",
    "configure_cache_paths",
    "configure_external_tool_paths",
    "ensure_cache_dirs",
    "resolve_cache_root",
]


def resolve_cache_root(cache_root: str | Path | None = None) -> Path:
    if cache_root is None:
        return PROJECT_ROOT / ".cache"
    path = Path(cache_root)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_cache_paths(
    cache_root: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    """读 config.yaml 的 cache.root（默认 .cache），算出各子目录路径"""
    global PROJECT_ROOT, CACHE_ROOT, HF_CACHE_DIR, HF_HUB_CACHE_DIR, TRANSFORMERS_CACHE_DIR
    global SENTENCE_TRANSFORMERS_CACHE_DIR, MODELSCOPE_CACHE_DIR, MINERU_CACHE_DIR, MINERU_OUTPUT_DIR
    global TORCH_CACHE_DIR, CHROMA_CACHE_DIR, UIE_CACHE_DIR
    global EXTERNAL_TOOLS_DIR, DEFAULT_TESSERACT_DIR, DEFAULT_POPPLER_DIR

    if project_root is not None:
        PROJECT_ROOT = Path(project_root).resolve()
    CACHE_ROOT = resolve_cache_root(cache_root)
    HF_CACHE_DIR = CACHE_ROOT / "huggingface"
    HF_HUB_CACHE_DIR = HF_CACHE_DIR / "hub"
    TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "transformers"
    SENTENCE_TRANSFORMERS_CACHE_DIR = HF_CACHE_DIR / "sentence_transformers"
    MODELSCOPE_CACHE_DIR = CACHE_ROOT / "modelscope"
    MINERU_CACHE_DIR = CACHE_ROOT / "mineru"
    MINERU_OUTPUT_DIR = MINERU_CACHE_DIR / "output"
    TORCH_CACHE_DIR = CACHE_ROOT / "torch"
    CHROMA_CACHE_DIR = CACHE_ROOT / "chroma"
    UIE_CACHE_DIR = CACHE_ROOT / "uie"
    EXTERNAL_TOOLS_DIR = CACHE_ROOT / "external"
    DEFAULT_TESSERACT_DIR = EXTERNAL_TOOLS_DIR / "ocr"
    DEFAULT_POPPLER_DIR = EXTERNAL_TOOLS_DIR / "poppler" / "poppler-24.08.0" / "Library" / "bin"
    return CACHE_ROOT


def _resolve_tool_dir(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def configure_external_tool_paths(
    *,
    tesseract_dir: str | Path | None = None,
    poppler_dir: str | Path | None = None,
) -> None:
    """配置 Tesseract / Poppler 目录（供 apply_cache_env 写入 PATH）。

    poppler_dir 省略时沿用系统 PATH（本机已在全局环境变量中配置 Poppler 时无需再设）。
    """
    global TESSERACT_DIR, POPPLER_DIR
    TESSERACT_DIR = _resolve_tool_dir(tesseract_dir, DEFAULT_TESSERACT_DIR)
    if poppler_dir is None or str(poppler_dir).strip() == "":
        POPPLER_DIR = None
    else:
        path = Path(poppler_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        POPPLER_DIR = path


def _prepend_path_dirs(*dirs: Path) -> None:
    parts = [str(path) for path in dirs if path.is_dir()]
    if not parts:
        return
    prefix = os.pathsep.join(parts)
    current = os.environ.get("PATH", "")
    if current:
        os.environ["PATH"] = prefix + os.pathsep + current
    else:
        os.environ["PATH"] = prefix


def _apply_external_tool_env() -> None:
    """将 OCR / Poppler 可执行文件目录加入 PATH（仅当目录存在时）。"""
    path_dirs: list[Path] = []
    if TESSERACT_DIR.is_dir():
        path_dirs.append(TESSERACT_DIR)
        tessdata = TESSERACT_DIR / "tessdata"
        if tessdata.is_dir():
            # Windows 便携版：语言包直接在 tessdata/ 下，PREFIX 指该目录
            os.environ["TESSDATA_PREFIX"] = str(tessdata)
        tesseract_exe = TESSERACT_DIR / "tesseract.exe"
        if tesseract_exe.is_file():
            os.environ["TESSERACT_CMD"] = str(tesseract_exe)
    if POPPLER_DIR is not None and POPPLER_DIR.is_dir():
        path_dirs.append(POPPLER_DIR)
    _prepend_path_dirs(*path_dirs)


def ensure_cache_dirs() -> None:
    """确保各缓存目录存在"""
    for path in (
        CACHE_ROOT,
        EXTERNAL_TOOLS_DIR,
        HF_CACHE_DIR,
        HF_HUB_CACHE_DIR,
        TRANSFORMERS_CACHE_DIR,
        SENTENCE_TRANSFORMERS_CACHE_DIR,
        MODELSCOPE_CACHE_DIR,
        MINERU_CACHE_DIR,
        MINERU_OUTPUT_DIR,
        TORCH_CACHE_DIR,
        CHROMA_CACHE_DIR,
        UIE_CACHE_DIR,
        HF_CACHE_DIR / "datasets",
        UIE_CACHE_DIR / "paddle",
    ):
        path.mkdir(parents=True, exist_ok=True)


def apply_cache_env() -> Path:
    """在 import HuggingFace / Torch 之前调用，将缓存定向到项目 .cache。

    强制覆盖系统/用户级 HF_HOME 等变量，避免其它项目路径抢占本项目 sentence_transformers 缓存。
    """
    global _CACHE_ENV_APPLIED
    ensure_cache_dirs()
    _apply_external_tool_env()
    os.environ["HF_HOME"] = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(TRANSFORMERS_CACHE_DIR)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(SENTENCE_TRANSFORMERS_CACHE_DIR)
    os.environ["MODELSCOPE_CACHE"] = str(MODELSCOPE_CACHE_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
    os.environ["PADDLE_HOME"] = str(UIE_CACHE_DIR / "paddle")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE_DIR)
    _CACHE_ENV_APPLIED = True
    return CACHE_ROOT


def cache_env_applied() -> bool:
    return _CACHE_ENV_APPLIED
