import torch
from torch import nn
import torch.nn.functional as F
from model.moe import SparseMoe
from model.model_config import AttentionConfig


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [1, *pos_cis.shape, 1]  # [1, seq_len, dim//2, 1]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, head_size: int):
        super(Attention, self).__init__()
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)
        self.head_size = head_size

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query = torch.reshape(query, [batch_size, seq_len, self.head_size, hidden_dim // self.head_size])
        key = torch.reshape(key, [batch_size, seq_len, self.head_size, hidden_dim // self.head_size])
        query, key = apply_rotary_emb(query, key, pos_cis)
        query = query.transpose(-2, -3)
        key = key.transpose(-2, -3)
        value = torch.reshape(value, [batch_size, seq_len, self.head_size, hidden_dim // self.head_size])
        value = value.transpose(-2, -3)
        attention_scores = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention_scores = attention_scores.transpose(-2, -3)
        attention_scores = torch.reshape(attention_scores, [batch_size, seq_len, hidden_dim])
        attention_scores = self.wo(attention_scores)
        return attention_scores


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, head_size: int, expert_num: int, share_num: int):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(hidden_dim, head_size)
        self.feed_forward = SparseMoe(expert_num, share_num, hidden_dim, hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pos_cis):
        h = x + self.attention(self.attention_norm(x), pos_cis)
        final_ret, expert_weights = self.feed_forward(self.ffn_norm(h))
        out = h + final_ret
        return out


class DouDiZhu(nn.Module):
    def __init__(self):
        super(DouDiZhu, self).__init__()
        self.cfg = AttentionConfig()

        self.tok_embeddings = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.dropout = nn.Dropout(0.01)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.cfg.attention_layer):
            self.layers.append(
                TransformerBlock(self.cfg.hidden_dim, self.cfg.head_num, self.cfg.expert_num, self.cfg.share_num))
        self.norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.output = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_size, bias=False)
        pos_cis = precompute_pos_cis(self.cfg.hidden_dim // self.cfg.head_num, self.cfg.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

    def forward(self, input_x: torch.Tensor):
        _, seq_len = input_x.shape
        h = self.tok_embeddings(input_x)
        h = self.dropout(h)
        pos_cis = self.pos_cis[0:seq_len]
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis)
        out = self.norm(h)
        return self.output(out)  # batch_size, seq_len, vocab_size
