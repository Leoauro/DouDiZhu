from dataclasses import dataclass


@dataclass
class AttentionConfig:
    head_num: int = 16
    attention_layer: int = 8
    hidden_dim: int = 512
    max_seq_len: int = 1024
    expert_num: int = 4  # 专家数量
    top_k: int = 2  # 每次选择的top_k 专家
    share_num: int = 2  # 共享专家数量
    vocab_size: int = 6400
