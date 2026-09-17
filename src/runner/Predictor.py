import sys
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

# 将 src 目录加入 sys.path，支持直接运行脚本
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config
from models.spell_check_bert import SpellCheckBert

_CHECKPOINT_PATH = config.CHECKPOINT_DIR / "spell_check_bert" / "best.pt"
_TOKENIZER_PATH = config.PRE_TRAINED_DIR / "bert-base-chinese"


class SpellCheckBertPredictor:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, inputs: list[str] | str):
        is_str = isinstance(inputs, str)
        if is_str:
            inputs = [inputs]
        # 处理输入数据
        encoded = self.tokenizer(
            inputs,
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        predictions = outputs['predictions']
        batch_result: list[str] = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        batch_result = [result.replace(' ', '') for result in batch_result]
        if is_str:
            return batch_result[0]
        return batch_result


@lru_cache
def get_spell_check_bert_predictor() -> SpellCheckBertPredictor:
    """懒加载 BERT 纠错模型（Agent Tool / 脚本共用）。"""
    if not _CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"BERT checkpoint 不存在: {_CHECKPOINT_PATH}")
    model = SpellCheckBert()
    model.load_state_dict(
        torch.load(_CHECKPOINT_PATH, map_location=torch.device("cpu"))
    )
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SpellCheckBertPredictor(model, tokenizer, device)


if __name__ == '__main__':
    predictor = get_spell_check_bert_predictor()
    print(
        predictor.predict(
            [
                '再加上在工作的地方有机会见面别的人，也可以学习新的文，最后经验越来越多。职业女生会增加她们的新智。',
                '我喜你。',
            ]
        )
    )
