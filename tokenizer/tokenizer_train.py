import json
import os
from itertools import islice
from tokenizers import Tokenizer, pre_tokenizers, models, trainers,decoders
from model.model_config import AttentionConfig


def data_dir():
    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

def read_texts_from_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            yield data['text']

if __name__ == '__main__':
    cfg = AttentionConfig()
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,  # 目标词表大小
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],  # 特殊Token
        show_progress=True,  # 显示进度条
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    print("初始化分词器完成")
    data_path = os.path.join(data_dir(), "data", "tokenizer_train.jsonl")
    texts = read_texts_from_jsonl(data_path)
    print("分词数据的前两条：")
    for text in islice(texts, 2):  # 从生成器中取前2条
        print(text)
    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # 保存词表和相关配置
    save_dir = os.path.join(data_dir(), "tokenizer", "tokenizer.json")
    tokenizer.save(save_dir)
    print("分词训练完成！")
