"""C 端细意图 BERT：other 保持 other，不并进 chitchat。"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

B2C_ROOT = Path(__file__).resolve().parents[1]
B2B_TRAIN = B2C_ROOT.parent / "B2B business" / "scripts" / "train_intent_classify_bert.py"

CONFIG = {
    "data_path": B2C_ROOT / "data" / "intent_classify" / "data.csv",
    "other_data_path": B2C_ROOT / "data" / "intent_classify" / "other_data.csv",
    "model_name": B2C_ROOT / "pretrained" / "bert-base-chinese",
    "output_dir": B2C_ROOT / "checkpoint" / "intent_classify",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "test_size": 0.2,
    "random_state": 42,
}


def load_and_preprocess_data(config: dict):
    df = pd.read_csv(config["data_path"], encoding="utf-8")
    other_path = config.get("other_data_path")
    if other_path and Path(other_path).exists():
        extra = pd.read_csv(other_path, encoding="utf-8")
        extra["intent"] = "other"
        df = pd.concat([df, extra], ignore_index=True)
        print(f"已合并 other 数据: {len(extra)} 条")
    df = df.dropna(subset=["text", "intent"]).reset_index(drop=True)
    print(f"有效数据: {len(df)} 条")
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["intent"])
    label_mapping = {int(i): str(label) for i, label in enumerate(encoder.classes_)}
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    train_df, test_df = train_test_split(
        df,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=df["label"] if len(df["label"].unique()) > 1 else None,
    )
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
    return train_df, test_df, encoder, label_mapping


def main() -> None:
    if not CONFIG["data_path"].exists():
        runpy.run_path(str(B2C_ROOT / "scripts" / "generate_intent_classify_data.py"), run_name="__main__")
    ns = runpy.run_path(str(B2B_TRAIN), run_name="b2b_intent_train")
    ns["load_and_preprocess_data"] = load_and_preprocess_data
    ns["train"](CONFIG)


if __name__ == "__main__":
    main()
