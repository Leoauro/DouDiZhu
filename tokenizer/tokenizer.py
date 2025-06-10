import os
from tokenizers import Tokenizer,decoders
from typing import List, Dict

class CustomTokenizer:
    def __init__(self):
        tokenizer_file = os.path.join(self._get_data_dir(), "tokenizer", "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"分词器文件未找到: {tokenizer_file}")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.special_tokens = {
            "unk": "[UNK]",
            "cls": "[CLS]",
            "sep": "[SEP]",
            "pad": "[PAD]",
            "mask": "[MASK]"
        }

    def _get_data_dir(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def detokenize(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        return text.strip()

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_token(self, name: str) -> str:
        return self.special_tokens.get(name.lower(), "[UNK]")

    def get_special_token_id(self, name: str) -> int:
        return self.tokenizer.encode(self.get_special_token(name)).ids[0]

    def apply_train_template(self, messages:List[Dict], thinking: bool):
        chatml_lines = ["[CLS]"]
        last_answer = ""
        if thinking:
            chatml_lines.append("<|thinking_tag_start|>thinking<|thinking_tag_end|>")
        else:
            chatml_lines.append("<|thinking_tag_start|><|thinking_tag_end|>")
        for index, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            if role == "assistant" and thinking and index == len(messages) - 1:
                content += f'<|thinking_start|>{msg.get("thinking", "")}<|thinking_end|>'
            if role == "assistant" and index == len(messages) - 1:
                content += f'<|answer_start|>{msg.get("answer", "")}<|answer_end|>'
            if index < len(messages) - 1:
                chatml_lines.append(f'<|im_start|>{role}{content}<|im_end|>')
            else:
                last_answer = f'<|im_start|>{role}{content}<|im_end|>[SEP]'
        question = "".join(chatml_lines)
        question_ids = self.tokenize(question)
        answer_ids = self.tokenize(last_answer)
        return question_ids, answer_ids

