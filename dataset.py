import numpy as np
from torch.utils.data import Dataset
import torch
from tokenizer.tokenizer import CustomTokenizer
import pandas as pd


class PretrainDataset(Dataset):
    def __init__(self, df, max_length=512):
        super().__init__()
        self.df = df
        self.tokenizer = CustomTokenizer()
        self.max_length = max_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        text = f"{'[CLS]'}{str(sample['text'])}{'[SEP]'}"
        input_id = self.tokenizer.tokenize(text)[:self.max_length]
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + self.tokenizer.tokenize("[PAD]") * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y)


class SFTDataset(Dataset):
    def __init__(self, df, max_length=1024):
        super().__init__()
        self.df = df
        self.max_length = max_length
        #
        self.tokenizer = CustomTokenizer()

    def __len__(self):
        return self.df.shape[0]

    def safe_eval(self, s):
        try:
            res = eval(s)
        except Exception as e:
            return []
        return res

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        history = self.safe_eval(sample['history'])
        q = str(sample['q'])
        a = str(sample['a'])

        messages = []
        for history_message in history:
            if len(history_message) <= 1:
                continue
            messages.append(
                {"role": 'user', "content": str(history_message[0])[:self.max_length // 2]}
            )
            messages.append(
                {"role": 'assistant', "content": str(history_message[1])[:self.max_length // 2]}
            )

        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        question_ids, answer_ids = self.tokenizer.apply_train_template(messages, False)
        if len(question_ids + answer_ids) > self.max_length:
            print("存在数据超过最长长度")
            question_ids, answer_ids = self.def_message()
        input_id = (question_ids + answer_ids)[:self.max_length]
        padding_len = self.max_length - len(input_id)
        input_id = input_id + self.tokenizer.tokenize("[PAD]") * padding_len

        mask_len = len(input_id) - len(question_ids) - padding_len
        # 0表示不计算损失
        loss_mask = [0] * len(question_ids) + [1] * (mask_len) + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)

        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        loss_mask_tensor = torch.from_numpy(loss_mask)
        return X_tensor, Y_tensor, loss_mask_tensor

    def def_message(self):
        messages = [
            {"role": "user", "content": "你是谁？"},
            {"role": "assistant", "content": "我是由谢正才研发的智能机器人，你需要什么帮助吗？"},
        ]
        question_ids, answer_ids = self.tokenizer.apply_train_template(messages, False)
        return question_ids, answer_ids


if __name__ == "__main__":
    df = pd.read_csv('./data/sft_data_single.csv')
    df = df.sample(frac=1.0)
    dataset = SFTDataset(df)
    print(dataset[0])
