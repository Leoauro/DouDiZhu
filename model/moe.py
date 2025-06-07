import torch
from torch import nn, Tensor
from torch.nn import functional as F

from model.expert import Expert
from model.model_config import AttentionConfig


class SparseMoe(nn.Module):
    def __init__(self, expert_num, share_num, in_feature, out_feature, hidden_dim):
        super(SparseMoe, self).__init__()
        self.cfg = AttentionConfig()
        self.out_feature = out_feature
        self.expert_list = nn.ModuleList(
            [
                Expert(
                    in_feature,
                    out_feature,
                    hidden_dim,
                ) for _ in range(expert_num)
            ]
        )

        self.share_expert = nn.ModuleList(
            [
                Expert(
                    in_feature,
                    out_feature,
                    hidden_dim,
                ) for _ in range(share_num)
            ]
        )

        self.gate = nn.Linear(in_feature, expert_num)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x shape (batch_size,seq_len,hidden_dim)
        # 由于每个专家的选中无法从x张量中抽离出新的张量，因为形状无法对齐， 先将x进行压平
        # flat_x shape(batch_size * seq_len,-1)
        flat_x = x.reshape(-1, x.size(-1))
        # 计算专家权重
        expert_weights = self.gate(flat_x)
        expert_weights = F.softmax(expert_weights, dim=-1)
        # topk_value, topk_idx shape (batch_size*seq_len,top_k)
        # topk_value 的值是专家权重， topk_idx 值是专家索引,索引我value都是经过排序的
        topk_value, topk_idx = expert_weights.topk(self.cfg.top_k, -1)
        # 重新初始化权重
        topk_value = topk_value / topk_value.sum(dim=-1, keepdim=True)

        # 由于专家索引在 topk_idx的value上，我们需要将其转换为索引上， 所以最好的办法是topk_idx的value进行one_hot 编码
        # 此时得到的 topk_idx shape (batch_size*seq_len,top_k,expert_num), 值为 0 和 1 ， 1 代表选中的专家,,expert_num 表示专家索引
        topk_idx = F.one_hot(topk_idx, num_classes=self.cfg.expert_num)

        # topk_idx shape(expert_num,top_k,batch_size*seq_len)
        topk_idx = topk_idx.permute(2, 1, 0)

        # 初始化最终结果张量
        final_ret = torch.zeros([flat_x.size(0), self.out_feature], device=x.device, dtype=x.dtype)
        # 接下来的目的，是找对应类别的专家，去执行他们需要处理的token，并计算权重
        for expert_idx in range(self.cfg.expert_num):
            cur_expert = self.expert_list[expert_idx]
            # cur shape (top_k,batch_size*seq_len)  其值为0 和 1 ，0表示被选中，， 1 表示未被选中
            cur = topk_idx[expert_idx]
            # selected_x 为查找非0的所有索引，
            # 其返回值的元素个数等于cur_x 的维度，
            # 返回的第一个值代表第一个维度上的索引， 第二个值代表第二维度上的索引 ....
            # 每一个返回值都是一个元组， 元组的长度都相同
            selected_topk_idx, selected_token_idx = torch.nonzero(cur, as_tuple=True)
            # cur_x  shape(selected_token_idx,hidden_dim)
            cur_x = flat_x[selected_token_idx, :]
            # cur_weight shape (selected_tokens) 值为权重
            cur_weight = topk_value[selected_token_idx, selected_topk_idx]
            cur_weight = cur_weight.unsqueeze(dim=-1)
            # 专家执行 expert_ret shape(selected_token_idx,out_feature)
            expert_ret = cur_expert(cur_x)

            # 专家执行结果与权重进行计算
            # expert_ret shape(selected_token_idx,out_feature)
            expert_ret = expert_ret * cur_weight
            final_ret.index_add_(0, selected_token_idx, expert_ret)

        # reshape 到标准结果
        final_ret = final_ret.reshape(x.size(0), x.size(1), self.out_feature)
        # 计算共享专家的结果
        for share_expert in self.share_expert:
            final_ret = final_ret + share_expert(x)

        return final_ret, expert_weights


