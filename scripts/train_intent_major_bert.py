"""C 端七类大意图 BERT：训练方式对齐老师 LabelEncoder + label_mapping.json。"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

B2C_ROOT = Path(__file__).resolve().parents[1]
B2B_TRAIN = B2C_ROOT.parent / "B2B business" / "scripts" / "train_intent_classify_bert.py"

CONFIG = {
    "data_path": B2C_ROOT / "data" / "intent_major" / "data.csv",
    "other_data_path": B2C_ROOT / "data" / "intent_major" / "other_data.csv",
    "model_name": B2C_ROOT / "pretrained" / "bert-base-chinese",
    "output_dir": B2C_ROOT / "checkpoint" / "intent_major",
    "max_length": 64,
    "batch_size": 16,
    "epochs": 4,
    "learning_rate": 2e-5,
    "test_size": 0.2,
    "random_state": 42,
}


def main() -> None:
    if str(B2C_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(B2C_ROOT / "scripts"))
    from train_intent_classify_bert import load_and_preprocess_data

    runpy.run_path(str(B2C_ROOT / "scripts" / "generate_intent_major_data.py"), run_name="__main__")

    import importlib.util

    spec = importlib.util.spec_from_file_location("b2b_intent_train_mod", B2B_TRAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载训练脚本: {B2B_TRAIN}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.load_and_preprocess_data = load_and_preprocess_data
    mod.train(CONFIG)


if __name__ == "__main__":
    main()
